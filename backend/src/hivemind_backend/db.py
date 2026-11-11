"""connection settings, read from the environment (see example.env)."""

import os

import psycopg
from psycopg.conninfo import make_conninfo

ROLES = {
    "backend": ("hm_backend", "HM_BACKEND_PASSWORD"),
    "agent": ("hm_agent", "HM_AGENT_PASSWORD"),
    "eval": ("hm_eval", "HM_EVAL_PASSWORD"),
}


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; copy example.env to .env and run with `uv run --env-file .env ...`")
    return value


def database_name() -> str:
    # HM_DB_NAME lets the tests use their own database
    return os.environ.get("HM_DB_NAME") or _env("POSTGRES_DB")


def conninfo(role: str = "admin", dbname: str | None = None) -> str:
    if role == "admin":
        user, pw = _env("POSTGRES_USER"), _env("POSTGRES_PASSWORD")
    else:
        user, pw_var = ROLES[role]
        pw = _env(pw_var)
    return make_conninfo(
        host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        port=os.environ.get("POSTGRES_PORT", "5433"),
        user=user,
        password=pw,
        dbname=dbname or database_name(),
        options="-c TimeZone=UTC",
    )


def connect(role: str = "admin", dbname: str | None = None, **kwargs) -> psycopg.Connection:
    return psycopg.connect(conninfo(role, dbname), **kwargs)
