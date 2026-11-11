"""shared test setup: a separate test database, migrated and loaded from a fixed seed."""

import os

import psycopg
import pytest

TEST_DB = "hivemind_test"
TEST_SEED = 42


@pytest.fixture(scope="session")
def test_db():
    """the dataset loaded into hivemind_test; HM_DB_NAME points the api at it."""
    from hivemind_backend.migrate import migrate
    from hivemind_datagen.dataset import generate
    from hivemind_datagen.generate import SMALL
    from hivemind_datagen.load import load

    try:
        migrate(TEST_DB, reset=True)
    except psycopg.OperationalError as exc:
        pytest.fail(f"postgres is not reachable ({exc}); start Docker Desktop, then `docker compose up -d`")
    tables = generate(TEST_SEED, SMALL)
    load(tables, TEST_DB)
    os.environ["HM_DB_NAME"] = TEST_DB
    return tables


@pytest.fixture(scope="session")
def test_db_name() -> str:
    return TEST_DB


@pytest.fixture(scope="session")
def test_seed() -> int:
    return TEST_SEED
