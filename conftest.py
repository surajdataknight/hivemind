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


# the cases a test opened: hivemind opens a case at the business's clock; the generated earlier claims are older
MADE_IN_TESTS = "select case_id from support.cases where created_at >= (select now from public.sim_clock)"


def undo_actions(conn) -> None:
    """takes back what the action gateway did in tests: its refunds (the generated ones have no key), their parts
    and ledger entries, the keys, the outbox, the decision records, and every order it cancelled."""
    made = "select refund_id from refunds_wallet.refunds where idempotency_key is not null"
    conn.execute(f"delete from refunds_wallet.refund_parts where refund_id in ({made})")
    conn.execute(f"delete from refunds_wallet.wallet_ledger where ref in ({made})")
    conn.execute("delete from refunds_wallet.refunds where idempotency_key is not null")
    conn.execute("truncate actions.idempotency_keys, notifications.notifications_outbox, support.decision_records")
    now = "(select now from public.sim_clock)"
    conn.execute(f"delete from orders.order_status_history where status = 'CANCELLED' and at >= {now}")
    conn.execute(
        "update orders.orders o set status = h.status, cancelled_at = null, version = o.version - 1 "
        "from (select distinct on (order_id) order_id, status from orders.order_status_history "
        "order by order_id, seq desc) h "
        f"where h.order_id = o.order_id and o.status = 'CANCELLED' and o.cancelled_at >= {now}")


@pytest.fixture
def clean_actions(test_db, test_db_name):
    from hivemind_backend import db

    with db.connect("admin", dbname=test_db_name) as conn:
        undo_actions(conn)
    yield
    with db.connect("admin", dbname=test_db_name) as conn:
        undo_actions(conn)


@pytest.fixture
def clean_chat_state(test_db, test_db_name):
    """no cases or chats left over from another test. the generated earlier claims stay: they are the customers'
    history."""
    from hivemind_backend import db

    with db.connect("admin", dbname=test_db_name) as conn:
        undo_actions(conn)
        conn.execute(f"delete from support.ticket_events where ticket_id in (select ticket_id from support.tickets "
                     f"where case_id in ({MADE_IN_TESTS}))")
        for table in ("support.tickets", "support.decision_records", "support.case_events"):
            conn.execute(f"delete from {table} where case_id in ({MADE_IN_TESTS})")
        conn.execute(f"delete from support.cases where case_id in ({MADE_IN_TESTS})")
        conn.execute("truncate agent.pii_vault, agent.messages, agent.conversations, agent.case_leases, "
                     "agent.evidence_ledger, agent.findings, agent.case_plans, agent.fact_sheets")
