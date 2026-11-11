"""creates the login roles and applies the numbered sql files in order, once each."""

import hashlib
from importlib import resources

import psycopg
from psycopg import sql

from hivemind_backend import db


def sql_files() -> list[tuple[str, str]]:
    folder = resources.files("hivemind_backend") / "sql"
    files = sorted((f.name, f.read_text(encoding="utf-8")) for f in folder.iterdir() if f.name.endswith(".sql"))
    return files


def ensure_roles(conn: psycopg.Connection) -> None:
    for role, pw_var in db.ROLES.values():
        pw = db._env(pw_var)
        exists = conn.execute("select 1 from pg_roles where rolname = %s", (role,)).fetchone()
        verb = "alter" if exists else "create"
        conn.execute(sql.SQL(verb + " role {} login password {}").format(sql.Identifier(role), sql.Literal(pw)))


def create_database(dbname: str, drop_first: bool = False) -> None:
    with db.connect("admin", dbname="postgres", autocommit=True) as conn:
        if drop_first:
            conn.execute(sql.SQL("drop database if exists {} with (force)").format(sql.Identifier(dbname)))
        exists = conn.execute("select 1 from pg_database where datname = %s", (dbname,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("create database {}").format(sql.Identifier(dbname)))
        ensure_roles(conn)


def migrate(dbname: str | None = None, reset: bool = False) -> list[str]:
    """returns the names of the files applied in this run."""
    dbname = dbname or db.database_name()
    create_database(dbname, drop_first=reset)
    applied = []
    with db.connect("admin", dbname=dbname, autocommit=True) as conn:
        conn.execute(
            "create table if not exists public.schema_migrations "
            "(name text primary key, sha256 text not null, applied_at timestamptz not null default now())"
        )
        for name, text in sql_files():
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            row = conn.execute("select sha256 from public.schema_migrations where name = %s", (name,)).fetchone()
            if row:
                if row[0] != digest:
                    raise RuntimeError(f"{name} changed after it was applied; run `hm-db migrate --reset`")
                continue
            with conn.transaction():
                conn.execute(text)
                conn.execute("insert into public.schema_migrations (name, sha256) values (%s, %s)", (name, digest))
            applied.append(name)
    return applied
