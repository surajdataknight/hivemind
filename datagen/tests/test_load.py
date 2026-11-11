"""the same seed gives the same rows in postgres, row for row."""

import pytest
from psycopg import sql

from hivemind_backend import db
from hivemind_datagen.dataset import generate
from hivemind_datagen.generate import SMALL, TABLES
from hivemind_datagen.load import load

pytestmark = pytest.mark.db


def fingerprint(dbname: str) -> dict[str, str]:
    """md5 of every row's text form, per table, in a fixed order."""
    out = {}
    with db.connect("admin", dbname=dbname) as conn:
        for name in TABLES:
            table = sql.Identifier(*name.split("."))
            query = sql.SQL("select count(*), md5(coalesce(string_agg(t::text, '|' order by t::text), '')) "
                            "from {} t").format(table)
            count, digest = conn.execute(query).fetchone()
            out[name] = f"{count}:{digest}"
    return out


def test_same_seed_same_rows_in_postgres(test_db, test_db_name, test_seed):
    first = fingerprint(test_db_name)
    load(generate(test_seed + 1, SMALL), test_db_name)
    other = fingerprint(test_db_name)
    load(generate(test_seed, SMALL), test_db_name)
    again = fingerprint(test_db_name)
    assert again == first
    assert other != first
    assert all(not v.startswith("0:") for k, v in first.items()), "a table was left empty"
