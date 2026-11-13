"""the scenario panel's read-only api (phase 2.6): what hivemind did on each case, kept from customers by its own key.
it reads only hivemind's agent schema, never the business's tables or the hidden truth."""

import inspect
import re

import pytest
from fastapi.testclient import TestClient

from hivemind_agents import panel, settings
from hivemind_agents.api import create_app
from hivemind_agents.panel import PanelReader
from scripted import NO_EXTRA_CHECKS, draft, issue

pytestmark = pytest.mark.db

KEY = "k" * 40


@pytest.fixture
def api(service):
    return TestClient(create_app(service, PanelReader(settings.agent_conninfo()), KEY))


def as_panel(key=KEY) -> dict:
    return {"Authorization": f"Bearer {key}"}


def run_a_case(chat_as, data, model):
    fault = data.fault("missing_item", "store_miss")
    model.intakes.append(draft(issues=[issue("item_missing_or_wrong", order_id=fault["order_id"], items=["paneer"])]))
    model.intakes.append(NO_EXTRA_CHECKS)
    chat = chat_as(fault["customer_id"])
    chat.say(f"Paneer was missing from my order {fault['order_id']}.")
    return fault, chat


def test_the_panel_needs_its_key_and_a_customers_session_is_not_one(api, chat_as, data, model, backend_http):
    assert api.get("/v1/panel/cases").status_code == 401
    assert api.get("/v1/panel/cases", headers=as_panel("x" * 40)).status_code == 401
    _, chat = run_a_case(chat_as, data, model)
    assert api.get("/v1/panel/cases", headers={"Authorization": f"Bearer {chat.token}"}).status_code == 401


def test_with_no_key_set_the_panel_is_off(service):
    off = TestClient(create_app(service, PanelReader(settings.agent_conninfo()), None))
    response = off.get("/v1/panel/cases", headers=as_panel())
    assert response.status_code == 503 and "HM_PANEL_KEY" in response.json()["detail"]


def test_a_case_shows_its_plan_checks_findings_and_fact_sheet(api, chat_as, data, model, cases):
    fault, _ = run_a_case(chat_as, data, model)
    (case,) = cases()
    (row,) = api.get("/v1/panel/cases", headers=as_panel()).json()["cases"]
    assert (row["case_id"], row["case_type"], row["order_id"], row["customer_id"]) == (
        case["case_id"], "item_missing_or_wrong", fault["order_id"], fault["customer_id"])
    assert row["running"] is False and row["findings"] > 0 and row["verdict"] == "sufficient_with_uncertainty"

    shown = api.get(f"/v1/panel/cases/{case['case_id']}", headers=as_panel()).json()
    (run,) = shown["runs"]
    tasks = {t["task_id"] for t in run["plan"]["tasks"]}
    assert set(run["outcomes"]) == tasks and {o["status"] for o in run["outcomes"].values()} == {"done"}
    assert "findings" not in next(iter(run["outcomes"].values()))  # findings are listed once, below
    reads = {r["evidence_id"] for r in shown["reads"]}
    assert shown["findings"] and all(set(f["evidence_refs"]) <= reads for f in shown["findings"])
    (sheet,) = shown["fact_sheets"]
    assert sheet["verdict"] == "sufficient_with_uncertainty" and sheet["sheet"]["round"] == 1
    assert "payload" not in shown["reads"][0]  # which read, when; not what came back


def test_an_unknown_case_is_404(api):
    assert api.get("/v1/panel/cases/CASE-NOPE", headers=as_panel()).status_code == 404


def test_the_panel_reads_only_hiveminds_own_schema():
    """besides this, its database role (hm_agent) cannot read the business's schemas or the truth at all."""
    source = re.sub(r"^from .* import .*$", "", inspect.getsource(panel), flags=re.M)  # python's imports
    tables = set(re.findall(r"\b(?:from|join)\s+([a-z_]+)\.", source))
    assert tables == {"agent"}
