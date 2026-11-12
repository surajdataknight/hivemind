"""the full case manager's parts on their own: the run lease, the evidence ledger, the case board, and the order
in which the parts of one reply reach the screen."""

import operator
from datetime import timedelta
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from hivemind_agents import board, evidence, settings
from hivemind_agents.backend_client import BackendClient, BackendError
from hivemind_agents.capabilities import intake
from hivemind_agents.case_store import CaseStore, LeaseLost
from hivemind_agents.contracts import Finding, IntentHint, add_to_boards
from hivemind_agents.reply_stream import ReplyStream

# the run lease


@pytest.fixture
def store_for(clean_chat_state):
    opened: list[CaseStore] = []

    def make(holder: str, ttl: timedelta = timedelta(minutes=2)) -> CaseStore:
        store = CaseStore(settings.agent_conninfo(), holder=holder, ttl=ttl)
        opened.append(store)
        return store

    yield make
    for store in opened:
        store.close()


@pytest.mark.db
def test_one_run_holds_a_case_at_a_time(store_for):
    first, second = store_for("run-a"), store_for("run-b")
    lease = first.acquire("CASE-AAAAAAAAAAAA")
    assert lease.fence == 1 and first.acquire("CASE-AAAAAAAAAAAA") == lease  # holding it again changes nothing
    assert second.acquire("CASE-AAAAAAAAAAAA") is None
    assert second.acquire("CASE-BBBBBBBBBBBB").fence == 1  # another case is free
    first.release("CASE-AAAAAAAAAAAA")
    assert second.acquire("CASE-AAAAAAAAAAAA").fence == 2  # the number only grows


@pytest.mark.db
def test_a_lease_that_ran_out_is_taken_over_and_the_old_holder_can_no_longer_write(store_for):
    stalled = store_for("stalled-run", ttl=timedelta(0))  # its lease runs out at once
    old = stalled.acquire("CASE-CCCCCCCCCCCC")
    other = store_for("other-run")
    new = other.acquire("CASE-CCCCCCCCCCCC")
    assert new.fence == old.fence + 1

    with pytest.raises(LeaseLost):
        stalled.add_findings(old, [finding("CASE-CCCCCCCCCCCC", "ev_x")])
    stalled.release("CASE-CCCCCCCCCCCC")  # a late release by the old holder does not end the new lease
    assert store_for("third-run").acquire("CASE-CCCCCCCCCCCC") is None
    assert other.findings("CASE-CCCCCCCCCCCC") == []


@pytest.mark.db
def test_closing_a_run_lets_go_of_its_cases(store_for):
    run = store_for("run-a")
    run.acquire("CASE-DDDDDDDDDDDD")
    run.close()
    assert store_for("run-b").acquire("CASE-DDDDDDDDDDDD").fence == 2


# the evidence ledger and the board


def read(**body) -> dict:
    """a read's body as the business sends it, with its evidence id."""
    return {**body, "evidence_id": "ev_" + evidence.body_hash(body)[:32]}


def finding(case_id: str, *refs: str, **extra) -> Finding:
    return Finding(case_id=case_id, task_id="t1", capability="order_investigator", type="order.line_missing",
                   claim="butter is on the order", evidence_refs=list(refs), confidence="high", **extra)


@pytest.mark.db
def test_only_findings_proven_by_this_cases_reads_stay_on_the_board(store_for):
    run = store_for("run-a")
    run.acquire("CASE-EEEEEEEEEEEE")
    mine, theirs = read(order_id="ORD-100001"), read(order_id="ORD-100002")
    run.record_evidence("CASE-EEEEEEEEEEEE", "getOrder", mine)
    run.record_evidence("CASE-EEEEEEEEEEEE", "getOrder", mine)  # the same read twice: one entry
    run.record_evidence("CASE-FFFFFFFFFFFF", "getOrder", theirs)
    grounded = finding("CASE-EEEEEEEEEEEE", mine["evidence_id"])
    made_up = finding("CASE-EEEEEEEEEEEE", mine["evidence_id"], "ev_never_read")
    borrowed = finding("CASE-EEEEEEEEEEEE", theirs["evidence_id"])
    elsewhere = finding("CASE-FFFFFFFFFFFF", theirs["evidence_id"])

    posted = board.post("CASE-EEEEEEEEEEEE", [grounded, made_up, borrowed, elsewhere], run)
    assert posted.kept == [grounded]
    assert [(f, why.split(" ")[0]) for f, why in posted.dropped] == [
        (made_up, "no"), (borrowed, "no"), (elsewhere, "belongs")]
    assert run.findings("CASE-EEEEEEEEEEEE") == [grounded]
    assert run.evidence_ids("CASE-EEEEEEEEEEEE") == {mine["evidence_id"]}
    assert posted.update() == {"boards": {"CASE-EEEEEEEEEEEE": [grounded]}}


@pytest.mark.db
def test_a_run_without_the_lease_cannot_post(store_for):
    run = store_for("run-a")
    with pytest.raises(LeaseLost):
        board.post("CASE-GGGGGGGGGGGG", [], run)


def test_the_board_only_ever_grows():
    a, b, c = (finding("CASE-1", "ev_1") for _ in range(3))
    boards = add_to_boards({"CASE-1": [a]}, {"CASE-1": [b], "CASE-2": [c]})
    assert boards == {"CASE-1": [a, b], "CASE-2": [c]}
    assert add_to_boards(boards, {"CASE-1": [a]}) == boards  # a finding posted twice is kept once
    assert add_to_boards(boards, {}) == boards


def test_findings_posted_side_by_side_all_reach_the_board():
    """two investigators run side by side (langgraph fan-out); neither one's findings are lost (blueprint b.5)."""

    class State(TypedDict, total=False):
        boards: Annotated[dict[str, list[Finding]], add_to_boards]
        ran: Annotated[list[str], operator.add]

    def investigator(name):
        return lambda state: {"boards": {"CASE-1": [finding("CASE-1", f"ev_{name}")]}, "ran": [name]}

    graph = StateGraph(State)
    for name in ("order", "delivery", "payment"):
        graph.add_node(name, investigator(name))
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    result = graph.compile().invoke({"boards": {}})
    assert sorted(result["ran"]) == ["delivery", "order", "payment"]
    assert sorted(f.evidence_refs[0] for f in result["boards"]["CASE-1"]) == ["ev_delivery", "ev_order", "ev_payment"]


def test_a_finding_needs_proof_and_a_known_shape():
    with pytest.raises(ValueError):
        finding("CASE-1")  # no evidence at all
    with pytest.raises(ValueError):
        Finding(case_id="CASE-1", task_id="t1", capability="x", type="Not A Type", claim="c",
                evidence_refs=["ev_1"], confidence="high")


# the evidence check on every read


class Answer:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class OneAnswer:
    def __init__(self, body):
        self.body = body

    def request(self, method, url, **kwargs):
        return Answer(self.body)


def test_a_read_whose_evidence_id_does_not_match_its_body_is_refused(tracer):
    body = read(order_id="ORD-100001", status="DELIVERED")
    kept = []
    client = BackendClient(OneAnswer({**body, "status": "CANCELLED"}), "token", tracer,
                           ledger=lambda *args: kept.append(args))
    client.case_id = "CASE-1"
    with pytest.raises(BackendError, match="evidence id"):
        client.order("ORD-100001")
    assert kept == []


def test_reads_are_kept_as_proof_only_while_a_case_is_worked(tracer):
    body = read(order_id="ORD-100001")
    kept = []
    client = BackendClient(OneAnswer(body), "token", tracer, ledger=lambda *args: kept.append(args))
    client.order("ORD-100001")  # before any case: not proof for anything
    client.case_id = "CASE-1"
    client.order("ORD-100001")
    assert kept == [("CASE-1", "getOrder", body)]


# one reply, several parts, in order


def shown(events) -> str:
    text = ""
    for e in events:
        text = e["text"] if e["type"] == "replace" else text + e["text"]
    return text


def test_one_template_part_arrives_only_with_done():
    out, events = ReplyStream(), []
    out.part_done("Your order is on its way.")
    assert events == []


def test_one_written_part_streams_as_it_did_before():
    out, events = ReplyStream(), []
    write = out.writer(events.append)
    write({"type": "token", "text": "Hi. "})
    write({"type": "token", "text": "Bye."})
    out.part_done("Hi. Bye.")
    assert events == [{"type": "token", "text": "Hi. "}, {"type": "token", "text": "Bye."}]


def test_parts_reach_the_screen_in_order():
    out, events = ReplyStream(), []
    out.part_done("A template.")                    # waits until something streams
    write = out.writer(events.append)
    write({"type": "token", "text": "Written "})
    write({"type": "replace", "text": "Written again."})  # replaces its own part only
    out.part_done("Written again.")
    out.part_done("Another template.")
    out.finish(events.append, ["Which order?"])
    assert shown(events) == "A template.\n\nWritten again.\n\nAnother template.\n\nWhich order?"


def test_two_template_parts_arrive_together_at_the_end():
    out, events = ReplyStream(), []
    out.part_done("One.")
    out.part_done("Two.")
    out.finish(events.append, [])
    assert events == [{"type": "token", "text": "One.\n\nTwo."}]


# issues waiting behind a question


def test_an_answer_brings_the_issues_waiting_behind_the_question():
    pending = {"asks": "order", "case_type": "offers_coupons", "coupon_code": None,
               "queue": [{"case_type": "refund", "order_id": None, "items_mentioned": ["milk"],
                          "claimed_amount_paise": None, "coupon_code": None}]}
    typed = intake.from_pending(pending, "ORD-100001")
    pressed = intake.from_button(IntentHint(case_type="offers_coupons", order_id="ORD-100001"), pending)
    for result in (typed, pressed):
        assert [(i.case_type, i.order_id) for i in result.issues] == [("offers_coupons", "ORD-100001"),
                                                                     ("refund", None)]
        assert result.issues[1].items_mentioned == ["milk"]
    # a different button is a new question, not the answer
    other = intake.from_button(IntentHint(case_type="where_is_my_order"), pending)
    assert [i.case_type for i in other.issues] == ["where_is_my_order"]
