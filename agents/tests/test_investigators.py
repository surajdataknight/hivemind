"""l2 capability tests (blueprint 11.1): each investigator on the test database, one fault at a time, against the
hidden truth of that fault. every finding must be proven by a read the check made (evidence integrity). the
quarantined reader is the scripted model's faithful reading; its own tests are in test_quarantined_reader.py."""

from datetime import datetime, timedelta, timezone

import pytest

from hivemind_agents import settings
from hivemind_agents.backend_client import BackendClient
from hivemind_agents.capabilities import (delivery_investigator, history_analyst, inventory_investigator, items,
                                          order_investigator, payment_investigator, risk_investigator)
from hivemind_agents.capabilities.identity import IdentityVerifier
from hivemind_agents.capabilities.investigators import ALL
from hivemind_agents.capabilities.planner import MINIMUM
from hivemind_agents.capabilities.registry import CheckInput
from hivemind_agents.contracts import CaseRef, Finding, Issue, Task
from hivemind_agents.llm import ModelRouter
from scripted import ScriptedModel, label_calls

pytestmark = [pytest.mark.db]

CASE_ID = "CASE-00000000TEST"


class Reads:
    """the reads one check made, as the evidence ledger would keep them."""

    def __init__(self):
        self.ids: set[str] = set()
        self.bodies: list[dict] = []

    def __call__(self, case_id, operation, body):
        assert case_id == CASE_ID
        self.ids.add(body["evidence_id"])
        self.bodies.append(body)


@pytest.fixture
def run_check(backend_http, tracer):
    """runs one capability for a customer and an order; checks every finding is proven by its own reads."""

    def run(name, customer_id, order_id=None, mentioned=(), board=(), model=None, amount=None):
        verifier = IdentityVerifier(settings.signing_key("HM_SESSION_SECRET"),
                                    settings.signing_key("HM_CASE_TOKEN_SECRET"))
        _, token = verifier.mint_case_token(customer_id, "SES-000001", "conv_test")
        reads = Reads()
        backend = BackendClient(backend_http, token, tracer, ledger=reads).for_case(CASE_ID)
        model = model or ScriptedModel()
        llm = ModelRouter(settings.load_models(), tracer, {"google": lambda: model, "groq": lambda: model})
        issue = Issue(case_type="item_missing_or_wrong", order_id=order_id, items_mentioned=list(mentioned),
                      claimed_amount_paise=amount)
        case = CaseRef(case_id=CASE_ID, case_type=issue.case_type, order_id=order_id, state="INVESTIGATING",
                       opened=True)
        cap = ALL[name]
        params = {p: order_id for p in cap.params}
        task = Task(task_id="m1", capability=name, params=params, depends_on=[], mandatory=True)
        result = cap.run(task, CheckInput(case, backend, issue=issue, board=tuple(board), llm=llm))
        known = reads.ids | {ref for f in board for ref in f.evidence_refs}
        for f in result.findings:
            assert f.case_id == CASE_ID and f.task_id == "m1" and f.capability == name
            assert set(f.evidence_refs) <= known, (f.type, f.evidence_refs)  # evidence integrity
        return result.findings, reads

    return run


def of_type(findings: list[Finding], type_: str) -> list[Finding]:
    return [f for f in findings if f.type == type_]


def line_of(data, order_id, sku):
    return next(line for line in data.t["orders.order_lines"] if line["order_id"] == order_id and line["sku"] == sku)


def words_for(line: dict) -> str:
    """how a customer names an item: its product name without the made-up brand or the pack size."""
    name = line["name"].split("(")[0].split(" ", 1)[1]
    return name.lower()


# the registry and the plan


def test_every_minimum_check_is_registered_with_the_scopes_the_token_carries():
    from hivemind_agents.capabilities.identity import SCOPES

    for case_type, names in MINIMUM.items():
        for name in names:
            assert name in ALL, (case_type, name)
            assert ALL[name].scopes <= set(SCOPES), name
            assert ALL[name].read_only


# c8 order


def test_order_names_the_claimed_lines_and_their_value(run_check, data):
    fault = data.fault("missing_item", "store_miss")
    sku = fault["claim"]["skus"][0]
    line = line_of(data, fault["order_id"], sku)
    found, _ = run_check("order", fault["customer_id"], fault["order_id"], mentioned=[words_for(line)])
    (status,) = of_type(found, "order.status")
    assert status.facts["status"] == "DELIVERED" and len(status.facts["lines"]) >= 2
    (claimed,) = of_type(found, "order.claimed_lines")
    assert [x["sku"] for x in claimed.facts["lines"]] == [sku]
    assert claimed.facts["claim_value_paise"] == line["qty"] * line["unit_paise"] == fault["expected"]["owed_paise"]


def test_order_says_when_a_named_item_is_not_on_the_order(run_check, data):
    fault = data.fault("missing_item", "false_claim")
    found, _ = run_check("order", fault["customer_id"], fault["order_id"], mentioned=["unicorn steak"])
    assert not of_type(found, "order.claimed_lines")
    (absent,) = of_type(found, "order.claim_not_on_order")
    assert absent.facts["mentions"] == ["unicorn steak"]


# c11 inventory & store


def test_store_miss_shows_unscanned_item_and_stock_out(run_check, data):
    fault = data.fault("missing_item", "store_miss")
    (missing,) = fault["facts"]["missing"]
    line = line_of(data, fault["order_id"], missing["sku"])
    found, _ = run_check("inventory_store", fault["customer_id"], fault["order_id"], mentioned=[words_for(line)])
    assert [f.facts["sku"] for f in of_type(found, "store.item_not_scanned")] == [missing["sku"]]
    (out,) = of_type(found, "store.stock_out_before_pack")
    assert out.facts["sku"] == missing["sku"] and 0 <= out.facts["minutes_before_pack"] <= 30
    assert not of_type(found, "store.item_scanned")


def test_a_false_claim_shows_the_item_packed(run_check, data):
    fault = data.fault("missing_item", "false_claim")
    line = line_of(data, fault["order_id"], fault["claim"]["skus"][0])
    found, _ = run_check("inventory_store", fault["customer_id"], fault["order_id"], mentioned=[words_for(line)])
    assert not of_type(found, "store.item_not_scanned")
    (scanned,) = [f for f in of_type(found, "store.item_scanned") if f.facts["sku"] == line["sku"]]
    assert scanned.facts["bag"] >= 1
    assert of_type(found, "store.bags_packed")[0].facts["bags"] >= 1


@pytest.mark.parametrize("variant", ["item_missing", "item_delivered"])
def test_conflicting_records_show_both_the_scan_and_the_incident(run_check, data, variant):
    fault = data.fault("conflicting_records", variant)
    sku = fault["facts"]["sku"]
    line = line_of(data, fault["order_id"], sku)
    model = ScriptedModel()
    found, _ = run_check("inventory_store", fault["customer_id"], fault["order_id"], mentioned=[words_for(line)],
                         model=model)
    assert [f.facts["sku"] for f in of_type(found, "store.item_scanned")] == [sku]
    incidents = [f for f in of_type(found, "store.incident") if f.facts["sku"] == sku]
    assert incidents and incidents[0].facts["type"] == "out_of_stock"
    assert incidents[0].facts["note_label"] == "note_mentions_out_of_stock"
    assert label_calls(model) == len(of_type(found, "store.incident"))


# c9 delivery


def test_rider_handover_shows_a_bag_short(run_check, data):
    fault = data.fault("missing_item", "rider_handover")
    found, _ = run_check("delivery", fault["customer_id"], fault["order_id"])
    store, _ = run_check("inventory_store", fault["customer_id"], fault["order_id"])
    (handover,) = of_type(found, "delivery.handover")
    (bags,) = of_type(store, "store.bags_packed")
    assert handover.facts["bags_handed"] == bags.facts["bags"] - 1


def test_rider_delay_shows_late_and_standing_still(run_check, data):
    fault = data.fault("delay", "rider_delay")
    found, _ = run_check("delivery", fault["customer_id"], fault["order_id"])
    (late,) = of_type(found, "delivery.late")
    assert late.facts["late_by_min"] == fault["facts"]["late_by_min"]
    (stopped,) = of_type(found, "delivery.partner_stopped")
    assert stopped.facts["minutes"] >= 20


def test_store_delay_is_late_with_a_store_incident_and_no_stop(run_check, data):
    fault = data.fault("delay", "store_delay")
    found, _ = run_check("delivery", fault["customer_id"], fault["order_id"])
    assert of_type(found, "delivery.late") and not of_type(found, "delivery.partner_stopped")
    store, _ = run_check("inventory_store", fault["customer_id"], fault["order_id"])
    shortage = [f for f in of_type(store, "store.incident") if f.facts["type"] == "staff_shortage"]
    assert shortage and shortage[0].facts["note_label"] == "note_mentions_delay"


def test_an_on_time_delivery_is_not_late(run_check, data):
    order = next(o for o in data.orders if o["status"] == "DELIVERED" and o["order_id"] not in data.fault_orders
                 and o["delivered_at"] <= o["promised_by"])
    found, _ = run_check("delivery", order["customer_id"], order["order_id"])
    assert of_type(found, "delivery.trip")[0].facts["late_by_min"] == 0
    assert not of_type(found, "delivery.late") and not of_type(found, "delivery.partner_stopped")


def test_a_steering_note_becomes_a_label_and_nothing_else(run_check, data):
    """blueprint s25: a partner note that asks for a refund is labelled; its words go nowhere."""
    fault = data.fault("injected_note", "partner_note")
    note = next(n for n in data.t["delivery.partner_notes"] if n["note_id"] == fault["facts"]["note_id"])
    found, _ = run_check("delivery", fault["customer_id"], fault["order_id"])
    (labelled,) = [f for f in of_type(found, "delivery.partner_note") if f.facts["note_id"] == note["note_id"]]
    assert labelled.facts["label"] == "note_gives_instructions"
    for f in found:
        assert note["text"] not in f.claim and note["text"] not in str(f.facts)


def test_an_unreadable_note_counts_for_nothing(run_check, data):
    fault = data.fault("injected_note", "partner_note")
    model = ScriptedModel()
    model.down = True
    found, _ = run_check("delivery", fault["customer_id"], fault["order_id"], model=model)
    notes = of_type(found, "delivery.partner_note")
    assert notes and all(f.facts["label"] == "unreadable" and f.confidence == "low" for f in notes)


# c10 payment


def test_duplicate_capture_is_found_with_the_extra_amount(run_check, data):
    fault = data.fault("duplicate_capture", "duplicate")
    found, _ = run_check("payment", fault["customer_id"], fault["order_id"])
    (dup,) = of_type(found, "payment.duplicate_capture")
    assert dup.facts["extra_paise"] == fault["expected"]["owed_paise"]
    assert fault["facts"]["capture_id"] in dup.facts["capture_ids"]


def test_failed_debit_not_reversed_is_found_with_its_due_date(run_check, data):
    fault = data.fault("failed_debit_not_reversed", "not_reversed")
    found, _ = run_check("payment", fault["customer_id"], fault["order_id"])
    (debit,) = of_type(found, "payment.failed_debit_not_reversed")
    assert debit.facts["intent_id"] == fault["facts"]["intent_id"]
    assert debit.facts["amount_paise"] == fault["expected"]["owed_paise"]
    debited = datetime.fromisoformat(debit.facts["debited_at"].replace("Z", "+00:00"))
    due = datetime.fromisoformat(debit.facts["reversal_due_by"].replace("Z", "+00:00"))
    assert due - debited in (timedelta(days=1), timedelta(days=5))
    as_of = data.t["truth.generation_runs"][0]["generated_at"]
    assert debit.facts["overdue"] == (as_of > due)


def test_a_plain_order_has_one_capture_and_nothing_else(run_check, data):
    order = next(o for o in data.orders if o["status"] == "DELIVERED" and o["order_id"] not in data.fault_orders)
    found, _ = run_check("payment", order["customer_id"], order["order_id"])
    (captured,) = of_type(found, "payment.captured")
    assert captured.facts["captured_paise"] == order["total_paise"]
    assert not of_type(found, "payment.duplicate_capture") and not of_type(found, "payment.failed_debit_not_reversed")


def test_earlier_refunds_are_reported(run_check, data):
    refund = next(r for r in data.t["refunds_wallet.refunds"] if r["case_id"])
    order = data.order(refund["order_id"])
    found, _ = run_check("payment", order["customer_id"], order["order_id"])
    (made,) = of_type(found, "payment.refunds_made")
    assert refund["refund_id"] in [r["refund_id"] for r in made.facts["refunds"]]


# c14 customer history and c15 risk


def ring(data, n=0) -> dict:
    return data.t["truth.fraud_rings"][n]


def quiet_customer(data) -> str:
    """a regular customer with delivered orders, no earlier claim and no shared device, address or upi id."""
    members = {m for r in data.t["truth.fraud_rings"] for m in r["members"]}
    claimed = {c["customer_id"] for c in data.t["support.cases"]}
    return next(c for c in sorted(data.by_customer) if c not in members and c not in claimed
                and sum(o["status"] == "DELIVERED" for o in data.by_customer[c]) >= 2
                and any(x["customer_id"] == c and x["persona"] == "regular" for x in data.t["customer.customers"]))


def test_history_counts_the_customers_claims(run_check, data):
    member = ring(data)["members"][0]
    found, _ = run_check("customer_history", member)
    (profile,) = of_type(found, "history.profile")
    claims = [c for c in data.t["support.cases"] if c["customer_id"] == member]
    assert profile.facts["claims"] == len(claims)
    assert profile.facts["claim_ratio"] == history_analyst.ratio(len(claims), profile.facts["delivered_orders"])
    assert "customer_id" not in profile.facts


def risk_for(run_check, customer_id, order_id, mentioned):
    history, _ = run_check("customer_history", customer_id)
    order, _ = run_check("order", customer_id, order_id, mentioned=mentioned)
    found, reads = run_check("risk", customer_id, order_id, board=history + order)
    (assessment,) = of_type(found, "risk.assessment")
    return assessment, reads


def test_a_ring_members_false_claim_is_high_risk_and_goes_to_a_person(run_check, data):
    """blueprint s09: scanned, handed over with the otp, a ring member with many claims: high risk, human review."""
    fault = data.fault("fraud_ring", "false_claim")
    line = line_of(data, fault["order_id"], fault["claim"]["skus"][0])
    assessment, _ = risk_for(run_check, fault["customer_id"], fault["order_id"], [words_for(line)])
    assert assessment.facts["band"] == "high" and assessment.facts["handling"] == "human_review"
    assert "linked_accounts_with_claims" in assessment.facts["signals"]
    assert assessment.facts["handling"] != "deny"


def test_a_quiet_customer_is_low_risk(run_check, data):
    customer = quiet_customer(data)
    order = next(o for o in data.by_customer[customer] if o["status"] == "DELIVERED")
    line = next(x for x in data.t["orders.order_lines"] if x["order_id"] == order["order_id"])
    assessment, _ = risk_for(run_check, customer, order["order_id"], [words_for(line)])
    assert assessment.facts["band"] == "low" and assessment.facts["handling"] == "proceed"


def test_risk_reads_the_history_itself_when_it_is_not_on_the_board(run_check, data):
    member = ring(data)["members"][0]
    found, reads = run_check("risk", member)
    assert of_type(found, "risk.assessment")
    assert {b["evidence_id"] for b in reads.bodies if "claims" in b}  # the profile was read for the proof


def test_the_risk_rules_on_the_blueprints_two_customers():
    """blueprint 3.3 (41 orders, 1 earlier claim, no links: low) and 3.5 (9 orders, 5 claims, 2 accounts on one
    device with claims, ₹1,850 of ₹2,100: high)."""
    calm = {"claims": 1, "claims_last_30_days": 0, "delivered_orders": 41, "account_age_days": 400,
            "escalations_rejected": 0}
    no_links = [{"link_type": t, "accounts": 0, "accounts_with_claims": 0, "claims": 0}
                for t in ("device", "address", "upi")]
    assert risk_investigator.assess(calm, no_links, 15_700, 124_000)[1] == "low"
    busy = {"claims": 5, "claims_last_30_days": 4, "delivered_orders": 9, "account_age_days": 200,
            "escalations_rejected": 0}
    device = [{"link_type": "device", "accounts": 2, "accounts_with_claims": 2, "claims": 6}] + no_links[1:]
    score, band, signals = risk_investigator.assess(busy, device, 185_000, 210_000)
    assert band == "high" and score == 1.0
    assert set(signals) == {"claim_velocity", "high_claim_ratio", "high_claim_value_share",
                            "linked_accounts_with_claims"}


def test_a_new_account_with_a_big_claim_is_not_low():
    young = {"claims": 0, "claims_last_30_days": 0, "delivered_orders": 1, "account_age_days": 2,
             "escalations_rejected": 0}
    links = [{"link_type": t, "accounts": 0, "accounts_with_claims": 0, "claims": 0}
             for t in ("device", "address", "upi")]
    assert risk_investigator.assess(young, links, 60_000, 200_000)[1] == "elevated"


# the pieces of plain code


@pytest.mark.parametrize("mention,expected", [
    ("paneer", ["paneer"]), ("the amul butter", ["butter"]), ("buter", ["butter"]), ("milk", ["toned", "cream"]),
    ("eggs", ["eggs"]), ("chocolate", []),
])
def test_item_words_match_order_lines(mention, expected):
    lines = [{"line_no": 1, "name": "Gokul Fresh Paneer (200 g)"}, {"line_no": 2, "name": "Nandi Salted Butter (100 g)"},
             {"line_no": 3, "name": "Gokul Toned Milk (500 ml)"}, {"line_no": 4, "name": "Nandi Full Cream Milk (1 l)"},
             {"line_no": 5, "name": "Farm Eggs (6 pcs)"}]
    named, _, unclear = items.claimed_lines([mention], lines)
    words = [next(w for w in ("paneer", "butter", "toned", "cream", "eggs") if w in line["name"].lower())
             for line in named]
    assert words == expected
    assert unclear == (len(expected) > 1)


def test_two_captures_of_one_amount_minutes_apart_are_a_duplicate():
    def cap(i, minute, amount=50_000, ref=None):
        at = datetime(2026, 10, 2, 10, minute, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"capture_id": f"CAP-{i}", "amount_paise": amount, "gateway_ref": ref or f"gw_{i}", "captured_at": at}

    assert len(payment_investigator.duplicates([cap(1, 0), cap(2, 1)])) == 1
    assert payment_investigator.duplicates([cap(1, 0), cap(2, 30)]) == []           # far apart
    assert payment_investigator.duplicates([cap(1, 0), cap(2, 1, 40_000)]) == []    # other amount
    assert payment_investigator.duplicates([cap(1, 0, ref="gw"), cap(2, 1, ref="gw")]) == []  # one gateway charge


def test_standing_still_is_measured_from_the_gps_points():
    start = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)

    def point(minute, lat):
        return {"at": (start + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ"), "lat": lat, "lng": 77.6}

    moving = [point(m, 12.90 + m * 0.003) for m in range(10)]
    assert delivery_investigator.longest_stop(moving) is None
    still = moving[:3] + [point(3 + m, 12.906 + (m % 2) * 0.0002) for m in range(15)]
    minutes, _, _ = delivery_investigator.longest_stop(still)
    assert minutes == 15  # from minute 2, where the partner reached the spot, to minute 17


def test_inventory_reads_stock_only_for_the_named_or_unscanned_items():
    assert inventory_investigator.MAX_SKUS == 6
    assert order_investigator.CAPABILITY.cost == "low" and inventory_investigator.CAPABILITY.cost == "medium"
