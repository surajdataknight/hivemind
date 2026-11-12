"""writes a generated dataset into postgres, replacing whatever data was there."""

from importlib import resources

from psycopg import sql
from psycopg.types.json import Jsonb

from hivemind_backend import db
from hivemind_datagen.generate import TABLES


def _value(v):
    return Jsonb(v) if isinstance(v, (dict, list)) else v


BUSINESS_SCHEMAS = ["identity", "customer", "catalog", "store_inventory", "orders", "payments", "refunds_wallet",
                    "delivery", "promotions", "risk", "support", "actions", "notifications"]


def load(tables: dict[str, list[dict]], dbname: str | None = None) -> None:
    """empties every business and truth table (cases and tickets too, since they point at the old orders),
    then copies the new rows in. the agent schema is left alone."""
    truth_sql = (resources.files("hivemind_datagen") / "truth.sql").read_text(encoding="utf-8")
    with db.connect("admin", dbname=dbname) as conn:
        if conn.execute("select to_regclass('orders.orders')").fetchone()[0] is None:
            raise RuntimeError("the business schemas are missing; run `hm-db migrate` first")
        conn.execute(truth_sql)
        existing = conn.execute(
            "select table_schema, table_name from information_schema.tables "
            "where table_type = 'BASE TABLE' and table_schema = any(%s) order by 1, 2",
            (BUSINESS_SCHEMAS + ["truth"],),
        ).fetchall()
        names = sql.SQL(", ").join(sql.Identifier(schema, table) for schema, table in existing)
        conn.execute(sql.SQL("truncate {}").format(names))
        with conn.cursor() as cur:
            for name in TABLES:
                rows = tables[name]
                if not rows:
                    continue
                columns = list(rows[0])
                statement = sql.SQL("copy {} ({}) from stdin").format(
                    sql.Identifier(*name.split(".")), sql.SQL(", ").join(sql.Identifier(c) for c in columns))
                with cur.copy(statement) as copy:
                    for row in rows:
                        copy.write_row([_value(row[c]) for c in columns])
        # the business's clock stops where the data ends
        as_of = tables["truth.generation_runs"][0]["generated_at"]
        conn.execute("insert into public.sim_clock (only_row, now) values (true, %s) "
                     "on conflict (only_row) do update set now = excluded.now", (as_of,))
        conn.commit()
