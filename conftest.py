"""shared test setup: a separate test database, migrated and loaded from a fixed seed."""

import os

import psycopg
import pytest

TEST_DB = "hivemind_test"
TEST_SEED = 42

# the tests sign their own tokens; a real .env is not needed for these two
os.environ.setdefault("HM_SESSION_SECRET", "test-only-session-signing-key-0123456789")
os.environ.setdefault("HM_CASE_TOKEN_SECRET", "test-only-case-token-signing-key-0123456789")


def pytest_addoption(parser):
    parser.addoption("--live-llm", action="store_true", default=False,
                     help="the scripted chats use the real models: needs GEMINI_API_KEY, and sends the made-up "
                          "chat text to google")
    parser.addoption("--live-pause", type=float, default=5.0,
                     help="seconds to wait after each live chat, to stay inside the free per-minute limits")


@pytest.fixture(scope="session")
def live(request) -> bool:
    return request.config.getoption("--live-llm")


@pytest.fixture(scope="session")
def live_pause(request) -> float:
    return request.config.getoption("--live-pause")


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
