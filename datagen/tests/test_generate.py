"""the generator without a database: same seed same data, sums add up, and each fault is really in the records."""

from collections import defaultdict
from datetime import UTC, timedelta

import pytest

from hivemind_datagen.dataset import generate
from hivemind_datagen.generate import FAULTS, SMALL, Generator, dataset_sha256

SEED = 42


@pytest.fixture(scope="module")
def t():
    return generate(SEED, SMALL)


@pytest.fixture(scope="module")
def idx(t):
    """rows grouped by order id, and other lookups the checks need."""
    by_order = defaultdict(lambda: defaultdict(list))
    for table in ("orders.order_lines", "orders.order_status_history", "store_inventory.pick_pack_scans",
                  "payments.payment_intents", "delivery.trips", "delivery.pod", "promotions.redemptions",
                  "refunds_wallet.refunds", "delivery.partner_notes"):
        for row in t[table]:
            by_order[row["order_id"]][table].append(row)
    intents = {row["intent_id"]: row for row in t["payments.payment_intents"]}
    by_intent = defaultdict(lambda: defaultdict(list))
    for table in ("payments.captures", "payments.gateway_events"):
        for row in t[table]:
            by_intent[row["intent_id"]][table].append(row)
    gps = defaultdict(list)
    for row in t["delivery.gps_points"]:
        gps[row["trip_id"]].append(row)
    return {"order": {o["order_id"]: o for o in t["orders.orders"]}, "rows": by_order, "intents": intents,
            "by_intent": by_intent, "gps": gps, "offers": {o["code"]: o for o in t["promotions.offers"]}}


def faults_of(t, fault_type, variant=None):
    found = [f for f in t["truth.order_faults"] if f["fault_type"] == fault_type
             and (variant is None or f["variant"] == variant)]
    assert found, f"no {fault_type}/{variant} in the data"
    return found


# reproducibility


def test_same_seed_same_data(t):
    again = generate(SEED, SMALL)
    assert again == t
    assert dataset_sha256(again) == dataset_sha256(t)


def test_other_seed_other_data(t):
    assert dataset_sha256(generate(SEED + 1, SMALL)) != dataset_sha256(t)


def test_run_record_holds_the_hash(t):
    run = t["truth.generation_runs"][0]
    assert run["seed"] == SEED
    assert run["data_sha256"] == dataset_sha256(t)
    assert run["config"]["days"] == SMALL.days


# shape


def test_every_fault_variant_present(t):
    present = {(f["fault_type"], f["variant"]) for f in t["truth.order_faults"]}
    assert present == {(ft, v) for ft, variants in FAULTS.items() for v in variants}


def test_faults_hit_distinct_orders(t):
    orders = [f["order_id"] for f in t["truth.order_faults"]]
    assert len(orders) == len(set(orders))


@pytest.mark.parametrize("seed", range(25))
def test_times_are_utc_whole_seconds_and_not_after_as_of(seed):
    promises = {"promised_by", "eta_at", "expires_at", "valid_to"}  # may lie after as_of
    for name, rows in generate(seed, SMALL).items():
        for row in rows:
            for key, value in row.items():
                if hasattr(value, "tzinfo"):
                    assert value.tzinfo == UTC, (name, key)
                    assert value.microsecond == 0, (name, key)
                    if key not in promises:
                        assert value <= SMALL.as_of, (name, key, value)


def test_some_orders_are_on_their_way_at_as_of(t):
    live = [o for o in t["orders.orders"] if o["status"] not in ("DELIVERED", "CANCELLED")]
    assert live


def test_money_adds_up(t, idx):
    duplicate = {f["order_id"] for f in t["truth.order_faults"] if f["fault_type"] == "duplicate_capture"}
    for order in t["orders.orders"]:
        rows = idx["rows"][order["order_id"]]
        lines = rows["orders.order_lines"]
        assert order["item_total_paise"] == sum(line["qty"] * line["unit_paise"] for line in lines)
        assert order["total_paise"] == (order["item_total_paise"] + order["delivery_fee_paise"]
                                        - order["discount_paise"])
        paid = [i for i in rows["payments.payment_intents"] if i["status"] == "succeeded"]
        assert len(paid) == 1 and paid[0]["amount_paise"] == order["total_paise"]
        captured = sum(c["amount_paise"] for c in idx["by_intent"][paid[0]["intent_id"]]["payments.captures"])
        copies = 2 if order["order_id"] in duplicate else 1
        assert captured == copies * order["total_paise"]


def test_status_history_in_order(t, idx):
    for order in t["orders.orders"]:
        history = idx["rows"][order["order_id"]]["orders.order_status_history"]
        assert [h["at"] for h in history] == sorted(h["at"] for h in history)
        assert history[-1]["status"] == order["status"]


def test_delivered_orders_have_scans_trip_and_pod(t, idx):
    store_miss = {f["order_id"] for f in faults_of(t, "missing_item", "store_miss")}
    for order in t["orders.orders"]:
        if order["status"] != "DELIVERED":
            continue
        rows = idx["rows"][order["order_id"]]
        assert len(rows["delivery.trips"]) == 1 and len(rows["delivery.pod"]) == 1
        packed = {s["sku"] for s in rows["store_inventory.pick_pack_scans"] if s["event"] == "packed"}
        ordered = {line["sku"] for line in rows["orders.order_lines"]}
        if order["order_id"] not in store_miss:
            assert packed == ordered


def test_coupons_follow_the_terms_except_injected_faults(t, idx):
    faulty = {f["order_id"] for f in faults_of(t, "coupon_misapplied")}
    for red in t["promotions.redemptions"]:
        if red["order_id"] in faulty:
            continue
        order = idx["order"][red["order_id"]]
        due, reason = Generator.offer_discount(idx["offers"][red["code"]], order["item_total_paise"],
                                               order["delivery_fee_paise"], first_order=True, at=order["placed_at"])
        if red["status"] == "applied":
            assert red["discount_paise"] == due == order["discount_paise"]
        else:
            assert red["reason"]


# each fault is really in the records


def test_store_miss(t, idx):
    for f in faults_of(t, "missing_item", "store_miss"):
        rows = idx["rows"][f["order_id"]]
        order = idx["order"][f["order_id"]]
        (missing,) = f["facts"]["missing"]
        assert not [s for s in rows["store_inventory.pick_pack_scans"] if s["sku"] == missing["sku"]]
        packing = next(h["at"] for h in rows["orders.order_status_history"] if h["status"] == "PACKING")
        outs = [e for e in t["store_inventory.stock_events"] if e["store_id"] == order["store_id"]
                and e["sku"] == missing["sku"] and e["event_type"] == "stock_out"
                and packing - timedelta(minutes=5) <= e["at"] <= packing]
        assert outs
        line = next(line for line in rows["orders.order_lines"] if line["sku"] == missing["sku"])
        assert line["status"] == "ORDERED"
        assert f["expected"]["owed_paise"] == line["qty"] * line["unit_paise"]
        assert not rows["refunds_wallet.refunds"]


def test_rider_handover(t, idx):
    for f in faults_of(t, "missing_item", "rider_handover"):
        rows = idx["rows"][f["order_id"]]
        bags = {s["bag"] for s in rows["store_inventory.pick_pack_scans"] if s["event"] == "packed"}
        (pod,) = rows["delivery.pod"]
        assert pod["bags_handed"] == len(bags) - 1
        lost = {m["sku"] for m in f["facts"]["missing"]}
        lines = [line for line in rows["orders.order_lines"] if line["sku"] in lost]
        assert f["expected"]["owed_paise"] == sum(line["qty"] * line["unit_paise"] for line in lines) > 0


def test_false_claim(t, idx):
    for f in faults_of(t, "missing_item", "false_claim"):
        rows = idx["rows"][f["order_id"]]
        bags = {s["bag"] for s in rows["store_inventory.pick_pack_scans"] if s["event"] == "packed"}
        assert rows["delivery.pod"][0]["bags_handed"] == len(bags)
        assert f["expected"] == {"owed_paise": 0, "proven_by_records": False}
        assert set(f["claim"]["skus"]) <= {line["sku"] for line in rows["orders.order_lines"]}


def test_delays(t, idx):
    for f in faults_of(t, "delay"):
        order = idx["order"][f["order_id"]]
        late = (order["delivered_at"] - order["promised_by"]).total_seconds() / 60
        assert late >= 15 and round(late) == f["facts"]["late_by_min"]
        trip = idx["rows"][f["order_id"]]["delivery.trips"][0]
        assert trip["delivered_at"] == order["delivered_at"] == idx["rows"][f["order_id"]]["delivery.pod"][0][
            "delivered_at"]
        points = idx["gps"][trip["trip_id"]]
        assert points[0]["at"] == trip["picked_up_at"] and points[-1]["at"] == trip["delivered_at"]
    for f in faults_of(t, "delay", "store_delay"):
        order = idx["order"][f["order_id"]]
        assert [i for i in t["store_inventory.incidents"] if i["store_id"] == order["store_id"]
                and i["type"] == "staff_shortage" and abs(i["at"] - order["placed_at"]) < timedelta(minutes=10)]
    for f in faults_of(t, "delay", "rider_delay"):
        trip = idx["rows"][f["order_id"]]["delivery.trips"][0]
        points = idx["gps"][trip["trip_id"]]
        # the rider stands still for at least 20 minutes
        still = max(sum(1 for q in points if abs(q["lat"] - p["lat"]) < 0.001 and abs(q["lng"] - p["lng"]) < 0.001)
                    for p in points)
        assert still >= 20


def test_failed_debit_not_reversed(t, idx):
    for f in faults_of(t, "failed_debit_not_reversed"):
        intent = idx["intents"][f["facts"]["intent_id"]]
        events = {e["type"] for e in idx["by_intent"][intent["intent_id"]]["payments.gateway_events"]}
        assert intent["status"] == "failed" and events == {"debit_success"}
        assert f["expected"]["owed_paise"] == intent["amount_paise"]


def test_duplicate_capture(t, idx):
    for f in faults_of(t, "duplicate_capture"):
        paid = [i for i in idx["rows"][f["order_id"]]["payments.payment_intents"] if i["status"] == "succeeded"][0]
        captures = idx["by_intent"][paid["intent_id"]]["payments.captures"]
        assert len(captures) == 2 and len({c["gateway_ref"] for c in captures}) == 2
        assert f["expected"]["owed_paise"] == captures[1]["amount_paise"] == paid["amount_paise"]


def test_coupon_wrong_amount(t, idx):
    for f in faults_of(t, "coupon_misapplied", "wrong_amount"):
        order = idx["order"][f["order_id"]]
        (red,) = idx["rows"][f["order_id"]]["promotions.redemptions"]
        due, reason = Generator.offer_discount(idx["offers"][red["code"]], order["item_total_paise"],
                                               order["delivery_fee_paise"], True, order["placed_at"])
        assert reason is None and red["discount_paise"] == order["discount_paise"] < due
        assert f["expected"]["owed_paise"] == due - red["discount_paise"]


def test_coupon_wrongly_rejected(t, idx):
    for f in faults_of(t, "coupon_misapplied", "wrongly_rejected"):
        order = idx["order"][f["order_id"]]
        (red,) = idx["rows"][f["order_id"]]["promotions.redemptions"]
        offer = idx["offers"][red["code"]]
        assert red["status"] == "rejected" and "below minimum" in red["reason"]
        assert order["item_total_paise"] >= offer["min_cart_paise"]
        due, reason = Generator.offer_discount(offer, order["item_total_paise"], order["delivery_fee_paise"], False,
                                               order["placed_at"])
        assert reason is None and f["expected"]["owed_paise"] == due > 0


def test_conflicting_records(t, idx):
    for f in faults_of(t, "conflicting_records"):
        order = idx["order"][f["order_id"]]
        sku = f["facts"]["sku"]
        scans = [s for s in idx["rows"][f["order_id"]]["store_inventory.pick_pack_scans"] if s["sku"] == sku]
        picked = next(s for s in scans if s["event"] == "picked")
        assert any(s["event"] == "packed" for s in scans)
        assert [i for i in t["store_inventory.incidents"] if i["store_id"] == order["store_id"] and i["sku"] == sku
                and i["type"] == "out_of_stock" and picked["at"] - timedelta(minutes=10) <= i["at"] < picked["at"]]
        assert f["expected"]["proven_by_records"] is False
        assert (f["expected"]["owed_paise"] > 0) == (f["variant"] == "item_missing")


# fraud rings and earlier claims (phase 2.4)


def test_fraud_rings_share_their_link(t):
    rings = t["truth.fraud_rings"]
    assert {r["link_type"] for r in rings} == {"device", "address", "upi"}
    members = [m for r in rings for m in r["members"]]
    assert len(members) == len(set(members)) == SMALL.fraud_rings * SMALL.ring_size
    for ring in rings:
        if ring["link_type"] == "device":
            devices = {s["device_id"] for s in t["identity.sessions"] if s["customer_id"] in ring["members"]}
            assert devices == {ring["shared"]}
        elif ring["link_type"] == "address":
            home = next(a for a in t["customer.addresses"] if a["address_id"] == ring["shared"])
            for member in ring["members"]:
                assert [a for a in t["customer.addresses"] if a["customer_id"] == member
                        and (a["line1"], a["pincode"], a["lat"], a["lng"]) ==
                        (home["line1"], home["pincode"], home["lat"], home["lng"])]
        else:
            for member in ring["members"]:
                refs = {i["instrument_ref"] for i in t["payments.payment_intents"]
                        if i["customer_id"] == member and i["method"] == "upi"}
                assert refs == {ring["shared"]}
    # no account outside a ring shares its link
    outside = [s for s in t["identity.sessions"] if s["customer_id"] not in members]
    assert len({s["device_id"] for s in outside}) == len(outside)


def test_past_claims_are_closed_legal_and_never_on_a_faulty_order(t, idx):
    from hivemind_backend.cases import can_move

    faulty = {f["order_id"] for f in t["truth.order_faults"]}
    events = defaultdict(list)
    for e in t["support.case_events"]:
        events[e["case_id"]].append(e)
    assert t["support.cases"]
    for case in t["support.cases"]:
        order = idx["order"][case["order_id"]]
        assert case["order_id"] not in faulty
        assert order["customer_id"] == case["customer_id"] and order["status"] == "DELIVERED"
        assert case["state"] == "CLOSED" and order["delivered_at"] < case["created_at"] <= case["updated_at"]
        history = sorted(events[case["case_id"]], key=lambda e: e["seq"])
        assert history[0]["type"] == "opened" and history[-1]["to_state"] == "CLOSED"
        for before, after in zip(history, history[1:]):
            assert can_move(before["to_state"], after["to_state"]) and before["at"] <= after["at"]
            assert after["from_state"] == before["to_state"]
    claim_refunds = [r for r in t["refunds_wallet.refunds"] if r["case_id"]]
    assert claim_refunds and {r["case_id"] for r in claim_refunds} <= {c["case_id"] for c in t["support.cases"]}
    tickets = t["support.tickets"]
    assert {k["status"] for k in tickets} <= {"APPROVED", "REJECTED"}


def test_ring_members_claim_far_more_often(t):
    ring = {m for r in t["truth.fraud_rings"] for m in r["members"]}
    claims = defaultdict(int)
    for case in t["support.cases"]:
        claims[case["customer_id"]] += 1
    delivered = defaultdict(int)
    for order in t["orders.orders"]:
        if order["status"] == "DELIVERED":
            delivered[order["customer_id"]] += 1
    for member in ring:
        assert claims[member] / delivered[member] >= 0.3
    others = [c for c in delivered if c not in ring]
    assert sum(claims[c] for c in others) / sum(delivered[c] for c in others) < 0.1


def test_ring_false_claim_and_injected_note(t, idx):
    ring = {m: r["ring_id"] for r in t["truth.fraud_rings"] for m in r["members"]}
    for f in faults_of(t, "fraud_ring", "false_claim"):
        rows = idx["rows"][f["order_id"]]
        assert f["customer_id"] in ring and f["facts"]["ring_id"] == ring[f["customer_id"]]
        (pod,) = rows["delivery.pod"]
        bags = {s["bag"] for s in rows["store_inventory.pick_pack_scans"] if s["event"] == "packed"}
        assert pod["otp_verified"] and pod["bags_handed"] == len(bags)
        assert f["expected"] == {"owed_paise": 0, "proven_by_records": False}
    for f in faults_of(t, "injected_note", "partner_note"):
        notes = idx["rows"][f["order_id"]]["delivery.partner_notes"]
        assert any(n["note_id"] == f["facts"]["note_id"] for n in notes)
        assert f["customer_id"] not in ring  # an unproven claim never lands on a ring member by chance
