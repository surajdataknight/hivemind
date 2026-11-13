"""the evaluation's own tests run on the shared test database with a scripted model: they check the scenarios, the
harness and the grader, not how well a model does."""

import pytest
from psycopg.rows import dict_row

from hivemind_agents import settings, tracing
from hivemind_agents.llm import ModelRouter
from hivemind_backend import db
from hivemind_evals import harness, scenarios
from scripted import ScriptedModel


@pytest.fixture
def model():
    return ScriptedModel()


@pytest.fixture
def llm(model):
    return ModelRouter(settings.load_models(), tracing.Tracer(tracing.make_provider(None)),
                       {"google": lambda: model, "groq": lambda: model}, busy_waits=(), sleep=lambda s: None)


@pytest.fixture
def shop(test_db, test_db_name):
    return harness.business(test_db_name)


@pytest.fixture
def service(clean_chat_state, shop, llm):
    return harness.make_service(shop, llm)


@pytest.fixture(scope="session")
def small_set(test_db, test_db_name):
    with db.connect("eval", dbname=test_db_name, row_factory=dict_row) as conn:
        return {s.scenario_id: s for s in scenarios.load(conn)}


@pytest.fixture
def records(test_db_name):
    from hivemind_evals.grader import read_records

    def read(conversation_id):
        with db.connect("eval", dbname=test_db_name, row_factory=dict_row) as conn:
            return read_records(conn, conversation_id)

    return read


@pytest.fixture(scope="session")
def hard_set(test_db, test_db_name):
    with db.connect("eval", dbname=test_db_name, row_factory=dict_row) as conn:
        return {s.scenario_id: s for s in scenarios.load_hard(conn)}
