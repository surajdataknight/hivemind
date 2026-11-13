"""the action gateway (blueprint c22): the only code that changes money or an order for support. every action:

- comes with an idempotency key (`<case>:remedy:<n>`, `<case>:cancel`, `<case>:outcome:<n>`) and follows the
  blueprint's steps (5.6): a key seen with the same request and finished returns the stored answer; a key seen with
  another request is refused (422); the work and the key's answer are kept in one database step, so a retry after a
  lost answer finds the same refund, never a second one.
- runs the eligibility engine again on the same inputs and refuses anything over what it allows, or anything it
  says a person must approve (an approval token comes with the executive's console, phase 5).
- holds the case and the order while it works, and a part (an order line, an extra capture, a failed debit, an
  order) is paid at most once whichever case asks (a unique index). money is paid one action at a time: the daily
  budgets are read and spent in one step, so two actions at once cannot both use the same room under a budget.
- writes, in the same step, the money (refund, or a wallet credit with its two ledger entries), the outbox event
  (cloudevents 1.0) and the decision record, which points at the hash of the case's record before it.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from hivemind_backend import eligibility
from hivemind_backend.auth import Caller
from hivemind_backend.evidence import canonical_json

KEY_TTL = timedelta(days=30)
WORKING_STATES = ("DECIDING", "ACTIONING")  # a case support is deciding on, or acting on
SETTLES = {"source": "3-5 business days", "wallet": "right away"}
SOURCE = "commerce-sim"
BUDGET_LOCK = 3_000_001  # the advisory lock held while money is paid


class Refused(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def request_hash(action: str, money: dict) -> str:
    return hashlib.sha256(canonical_json({"action": action, **money})).hexdigest()


def once(conn: psycopg.Connection, key: str, action: str, money: dict, now: datetime,
         work: Callable[[], dict]) -> tuple[dict, bool]:
    """(the answer, whether it is a replay). the key, the work and the answer are one transaction."""
    digest = request_hash(action, money)
    with conn.transaction():
        # a second caller with the same key waits here until the first one's step is done
        mine = conn.execute(
            "insert into actions.idempotency_keys (key, request_hash, status, created_at, expires_at) "
            "values (%s, %s, 'IN_PROGRESS', %s, %s) on conflict (key) do nothing returning key",
            (key, digest, now, now + KEY_TTL)).fetchone()
        if mine is None:
            row = conn.execute("select request_hash, status, response from actions.idempotency_keys where key = %s",
                               (key,)).fetchone()
            if row["request_hash"] != digest:
                raise Refused(422, f"the idempotency key {key} was already used for another request")
            if row["status"] == "COMPLETED":
                return row["response"], True
            raise Refused(409, f"the request with key {key} is still being worked on; try again later")
        answer = work()
        conn.execute("update actions.idempotency_keys set status = 'COMPLETED', response = %s where key = %s",
                     (Jsonb(answer), key))
    return answer, False


# what every action holds and checks first


def hold_case(conn: psycopg.Connection, case_id: str, who: Caller) -> dict:
    case = conn.execute(
        "select case_id, customer_id, order_id, issue_type, state from support.cases where case_id = %s "
        "and customer_id = %s for update", (case_id, who.customer_id)).fetchone()
    if case is None:
        raise Refused(404, f"case {case_id} does not exist")
    return case


def working(case: dict) -> None:
    if case["state"] not in WORKING_STATES:
        raise Refused(409, f"case {case['case_id']} is {case['state']}; support acts only on a case it is deciding")


def hold_budgets(conn: psycopg.Connection) -> None:
    """until the step ends, no other action reads or spends the daily budgets (taken after the case and the order,
    always in that order, so two actions never wait on each other)."""
    conn.execute("select pg_advisory_xact_lock(%s)", (BUDGET_LOCK,))


def hold_order(conn: psycopg.Connection, order_id: str) -> dict:
    return conn.execute("select order_id, customer_id, status, version from orders.orders where order_id = %s "
                        "for update", (order_id,)).fetchone()


def eligible(conn: psycopg.Connection, case: dict, inputs: Any, now: datetime) -> dict[str, dict]:
    claims = [eligibility.Claim(c.part, c.status) for c in inputs.claims]
    result = eligibility.evaluate(conn, case, claims, inputs.risk_band, inputs.records_conflict, inputs.verdict, now)
    return {p["part"]: p for p in result["parts"]}


def check_parts(parts: list[Any], allowed: dict[str, dict], kind: str, refund_type: str) -> None:
    seen = set()
    for p in parts:
        if p.part in seen:
            raise Refused(422, f"{p.part} is named twice")
        seen.add(p.part)
        rule = allowed.get(p.part)
        if rule is None or rule["outcome"] != "remedy" or rule["kind"] != kind:
            why = rule["why"] if rule else "not part of this case's claim"
            raise Refused(403, f"the rules allow no {kind} for {p.part}: {why}")
        if refund_type not in rule["allowed"]:
            raise Refused(403, f"{p.part} may not be paid as {refund_type}")
        if rule["requires"] != "auto":
            raise Refused(403, f"{p.part} needs a person's approval: {rule['why']}")
        if p.amount_paise > rule["max_amount_paise"]:
            raise Refused(403, f"{p.part}: {p.amount_paise} paise is over the {rule['max_amount_paise']} the rules "
                               "allow")


# the records an action leaves


def outbox(conn: psycopg.Connection, module: str, type_: str, subject: str, data: dict, now: datetime,
           traceparent: str | None) -> None:
    conn.execute(
        "insert into notifications.notifications_outbox (id, source, type, subject, time, data, traceparent) "
        "values (%s, %s, %s, %s, %s, %s, %s)",
        ("evt_" + uuid4().hex, f"{SOURCE}/{module}", type_, subject, now, Jsonb(data), traceparent))


def keep_record(conn: psycopg.Connection, case_id: str, record: dict, now: datetime) -> str:
    """the decision record, chained to the case's record before it. call with the case row held."""
    prev = conn.execute("select seq, record_hash from support.decision_records where case_id = %s "
                        "order by seq desc limit 1", (case_id,)).fetchone()
    record_id = "DR-" + uuid4().hex[:12].upper()
    body = {**record, "decision_record_id": record_id, "case_id": case_id, "created_at": now.isoformat(),
            "prev_record_hash": prev["record_hash"] if prev else None}
    digest = "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()
    conn.execute(
        "insert into support.decision_records (decision_record_id, case_id, record, prev_record_hash, created_at, "
        "record_hash, seq) values (%s, %s, %s, %s, %s, %s, %s)",
        (record_id, case_id, Jsonb(body), body["prev_record_hash"], now, digest, (prev["seq"] if prev else 0) + 1))
    return record_id


def _pay(conn: psycopg.Connection, case: dict, parts: list[tuple[str, int, str]], destination: str, reason: str,
         key: str, now: datetime) -> dict:
    """one refund row for the parts (part, amount, basis); a wallet credit also gets its two ledger entries."""
    refund_id = "RF-" + uuid4().hex[:12].upper()
    amount = sum(a for _, a, _ in parts)
    if amount > eligibility.paid_for_order(conn, case["order_id"]):
        raise Refused(409, "this would refund more than was paid for the order")
    status = "COMPLETED" if destination == "wallet" else "INITIATED"
    conn.execute(
        "insert into refunds_wallet.refunds (refund_id, order_id, amount_paise, destination, status, reason, case_id, "
        "decision_record_id, created_at, customer_id, route, idempotency_key) "
        "values (%s, %s, %s, %s, %s, %s, %s, null, %s, %s, 'auto', %s)",
        (refund_id, case["order_id"], amount, destination, status, reason, case["case_id"], now,
         case["customer_id"], key))
    try:
        with conn.transaction():  # a savepoint: a part already paid is a refusal, not a server error
            for part, part_amount, basis in parts:
                conn.execute("insert into refunds_wallet.refund_parts (refund_id, order_id, part, amount_paise, basis) "
                             "values (%s, %s, %s, %s, %s)", (refund_id, case["order_id"], part, part_amount, basis))
    except psycopg.errors.UniqueViolation:
        raise Refused(409, "a part of this request was already refunded") from None
    if destination == "wallet":
        txn = "TXN-" + uuid4().hex[:12].upper()
        for account, signed in ((f"customer:{case['customer_id']}:wallet", amount), ("business:support_credits",
                                                                                       -amount)):
            conn.execute("insert into refunds_wallet.wallet_ledger (entry_id, txn_id, account, amount_paise, ref, "
                         "created_at) values (%s, %s, %s, %s, %s, %s)",
                         ("WL-" + uuid4().hex[:12].upper(), txn, account, signed, refund_id, now))
    return {"refund_id": refund_id, "order_id": case["order_id"], "amount_paise": amount, "destination": destination,
            "status": status, "expected_settlement": SETTLES[destination],
            "parts": [{"part": p, "amount_paise": a, "basis": b} for p, a, b in parts]}


def _link(conn: psycopg.Connection, refund_id: str, record_id: str) -> None:
    conn.execute("update refunds_wallet.refunds set decision_record_id = %s where refund_id = %s",
                 (record_id, refund_id))


# the actions


def refund(conn: psycopg.Connection, who: Caller, body: Any, key: str, destination: str, now: datetime,
           traceparent: str | None) -> tuple[dict, bool]:
    action = "wallet_credit" if destination == "wallet" else "refund_to_source"
    money = {"case_id": body.case_id, "destination": destination,
             "parts": sorted([p.part, p.amount_paise] for p in body.parts)}

    def work() -> dict:
        case = hold_case(conn, body.case_id, who)
        working(case)
        if case["order_id"] is None:
            raise Refused(409, "the case has no order")
        hold_order(conn, case["order_id"])
        hold_budgets(conn)
        allowed = eligible(conn, case, body.eligibility, now)
        check_parts(body.parts, allowed, "refund", action)
        made = _pay(conn, case, [(p.part, p.amount_paise, allowed[p.part]["basis"]) for p in body.parts],
                    destination, f"{case['issue_type']}:{action}", key, now)
        outbox(conn, "refunds_wallet", "wallet.credited" if destination == "wallet" else "refund.initiated",
               case["order_id"], {**made, "case_id": case["case_id"]}, now, traceparent)
        action_row = {"type": action, "refund_id": made["refund_id"], "amount_paise": made["amount_paise"],
                      "idempotency_key": key, "status": made["status"]}
        record_id = keep_record(conn, case["case_id"], {**body.decision, "actions": [action_row],
                                                        "rules_version": eligibility.RULES_VERSION}, now)
        _link(conn, made["refund_id"], record_id)
        return {**made, "case_id": case["case_id"], "decision_record_id": record_id, "idempotency_key": key,
                "rules_version": eligibility.RULES_VERSION, "created_at": now.isoformat()}

    return once(conn, key, action, money, now, work)


def cancel(conn: psycopg.Connection, who: Caller, body: Any, key: str, now: datetime,
           traceparent: str | None) -> tuple[dict, bool]:
    money = {"case_id": body.case_id}

    def work() -> dict:
        case = hold_case(conn, body.case_id, who)
        working(case)
        if case["issue_type"] != "cancellation" or case["order_id"] is None:
            raise Refused(409, "only a cancellation case on an order can cancel it")
        order = hold_order(conn, case["order_id"])
        hold_budgets(conn)
        allowed = eligible(conn, case, body.eligibility, now).get("order")
        if allowed is None or allowed["outcome"] != "remedy" or allowed["requires"] != "auto":
            raise Refused(409, f"order {order['order_id']} can no longer be cancelled by support: "
                               f"{allowed['why'] if allowed else 'no rule'}")
        conn.execute("update orders.orders set status = 'CANCELLED', cancelled_at = %s, version = version + 1 "
                     "where order_id = %s", (now, order["order_id"]))
        conn.execute("insert into orders.order_status_history (order_id, seq, status, at) select %s, "
                     "coalesce(max(seq), 0) + 1, 'CANCELLED', %s from orders.order_status_history where order_id = %s",
                     (order["order_id"], now, order["order_id"]))
        outbox(conn, "orders", "order.cancelled", order["order_id"],
               {"order_id": order["order_id"], "case_id": case["case_id"], "by": "support"}, now, traceparent)
        made = None
        if allowed["max_amount_paise"] > 0:
            made = _pay(conn, case, [("order", allowed["max_amount_paise"], "order_cancelled")], "source",
                        "order_cancelled", key, now)
            outbox(conn, "refunds_wallet", "refund.initiated", order["order_id"], {**made, "case_id": case["case_id"]},
                   now, traceparent)
        actions = [{"type": "cancel_order", "order_id": order["order_id"], "idempotency_key": key,
                    "status": "CANCELLED"}]
        if made:
            actions.append({"type": "refund_to_source", "refund_id": made["refund_id"],
                            "amount_paise": made["amount_paise"], "idempotency_key": key, "status": made["status"]})
        record_id = keep_record(conn, case["case_id"], {**body.decision, "actions": actions,
                                                        "rules_version": eligibility.RULES_VERSION}, now)
        if made:
            _link(conn, made["refund_id"], record_id)
        return {"case_id": case["case_id"], "order_id": order["order_id"], "status": "CANCELLED", "refund": made,
                "decision_record_id": record_id, "idempotency_key": key, "rules_version": eligibility.RULES_VERSION,
                "created_at": now.isoformat()}

    return once(conn, key, "cancel_order", money, now, work)


def record_outcome(conn: psycopg.Connection, who: Caller, body: Any, key: str, now: datetime) -> tuple[dict, bool]:
    """a decision that moves no money (a person decides, or nothing is owed) is kept like any other."""
    money = {"case_id": body.case_id, "outcome": body.outcome}

    def work() -> dict:
        case = hold_case(conn, body.case_id, who)
        working(case)
        record_id = keep_record(conn, case["case_id"], {**body.decision, "outcome": body.outcome, "actions": [],
                                                        "rules_version": eligibility.RULES_VERSION}, now)
        return {"case_id": case["case_id"], "outcome": body.outcome, "decision_record_id": record_id,
                "idempotency_key": key, "created_at": now.isoformat()}

    return once(conn, key, "record_outcome", money, now, work)


def record_chain_ok(records: list[dict]) -> bool:
    """each record's hash is the hash of its body, and each points at the one before it (for tests and audits)."""
    prev = None
    for row in records:
        body = row["record"]
        if body.get("prev_record_hash") != prev:
            return False
        digest = "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()
        if digest != row["record_hash"]:
            return False
        prev = digest
    return True

