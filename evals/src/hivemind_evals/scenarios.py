"""scenarios from the hidden truth: each one is a real fault of the generated data, the customer who has it, what
they will say, and what a correct investigation must show. read with the grader's role (hm_eval), the only one that
sees the truth.

the small set (record 014: runs after each change, once each): one scenario per fault kind phase 2 can investigate;
in the first missing-item scenario whose order its day alone names, the customer names it only by that day
("yesterday's order").

the hard set (the case-lead test, 2026-11-13): cases where the plan matters more. in three, the customer does not
name the missing item until asked which; in four, one customer with two real faults writes both in one message
(a several-issue case: both are graded). each fault is used once, so one database serves the whole set.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta

import psycopg

from hivemind_agents.capabilities.triage import FITS, RECENT_ORDERS
from hivemind_agents.wording import IST

# the case type each fault is reported as
CASE_TYPE = {
    "missing_item": "item_missing_or_wrong", "conflicting_records": "item_missing_or_wrong",
    "fraud_ring": "item_missing_or_wrong", "injected_note": "item_missing_or_wrong",
    "failed_debit_not_reversed": "payment_failed", "duplicate_capture": "payment_failed",
    "coupon_misapplied": "offers_coupons", "delay": "where_is_my_order",
}
SMALL = [("missing_item", "store_miss"), ("missing_item", "rider_handover"), ("missing_item", "false_claim"),
         ("fraud_ring", "false_claim"), ("injected_note", "partner_note"), ("conflicting_records", "item_missing"),
         ("conflicting_records", "item_delivered"), ("failed_debit_not_reversed", "not_reversed"),
         ("duplicate_capture", "duplicate"), ("coupon_misapplied", "wrong_amount"),
         ("coupon_misapplied", "wrongly_rejected")]
# the hard set: the customer names no item at first
HARD_NO_ITEM = [("missing_item", "false_claim"), ("missing_item", "rider_handover"),
                ("conflicting_records", "item_missing")]
HARD_PAIRS = 4  # customers with two faults, both of a kind hivemind investigates, written in one message
PERSONAS = ["calm and polite, writes full sentences",
            "upset and short, writes in lowercase with little punctuation",
            "chatty, writes in Hinglish (English with some Hindi words in Latin letters)"]
# how the customer asks for the money on a missing-item claim: natural words, not only "please check" (seen
# 2026-11-13: "please give my money back" was read as a second issue). the rest ask "please check"
ASKS_FOR = {("missing_item", "store_miss"): "money_back", ("missing_item", "rider_handover"): "wallet"}
ASK_WORDS = {"check": "Please check.", "money_back": "Please give my money back.",
             "wallet": "Please put the money in my HiveMart wallet."}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass
class Scenario:
    scenario_id: str
    fault_id: str
    fault_type: str
    variant: str
    customer_id: str
    customer_name: str
    order_id: str
    case_type: str
    summary: str                     # what the customer will say, in short (from the truth's claim)
    items: list[dict] = field(default_factory=list)  # sku, name: the items the customer says are missing
    code: str | None = None          # the coupon, for a coupon problem
    names_order_by: str = "id"       # "id", or a day word: "yesterday", "monday", "2026-10-09" ...
    persona: str = PERSONAS[0]
    owed_paise: int = 0
    proven_by_records: bool = True
    facts: dict = field(default_factory=dict)  # what really happened: for the grader only, never the customer
    names_items: bool = True         # false: the first message names no item; the items only when asked which
    also: "Scenario | None" = None   # a second problem of the same customer, written in the same first message
    asks_for: str = "check"          # "check", "money_back" or "wallet": how the customer asks for the money

    def parts(self) -> list["Scenario"]:
        return [self] + ([self.also] if self.also else [])

    def to_json(self) -> dict:
        return asdict(self)


def _day_word(day: date, today: date) -> str:
    back = (today - day).days
    if back in (0, 1, 2):
        return ("today", "yesterday", "day_before_yesterday")[back]
    return WEEKDAYS[day.weekday()] if back < 7 else day.isoformat()


def _days(order: dict) -> set[date]:
    return {at.astimezone(IST).date() for at in (order["placed_at"], order["delivered_at"]) if at is not None}


def _by_day(order_id: str, case_type: str, newest: list[dict], today: date) -> str | None:
    """the day word that names this order alone among the customer's newest orders this problem fits, or None."""
    fitting = [o for o in newest if FITS.get(case_type) is None or o["status"] in FITS[case_type]]
    target = next((o for o in fitting if o["order_id"] == order_id), None)
    if target is None:
        return None
    for day in sorted(_days(target)):
        if sum(day in _days(o) for o in fitting) == 1:
            return _day_word(day, today)
    return None


def load(conn: psycopg.Connection, only: list[tuple[str, str]] | None = None) -> list[Scenario]:
    """the small set from the truth in this database; deterministic for one seed."""
    now: datetime = conn.execute("select now from public.sim_clock").fetchone()["now"]
    today = now.astimezone(IST).date()
    faults = conn.execute("select * from truth.order_faults order by fault_id").fetchall()
    out: list[Scenario] = []
    wanted = list(only or SMALL)
    for n, (fault_type, variant) in enumerate(wanted):
        fault = next((f for f in faults if (f["fault_type"], f["variant"]) == (fault_type, variant)), None)
        if fault is None:
            continue
        out.append(_scenario(conn, fault, f"{fault_type}/{variant}", PERSONAS[n % len(PERSONAS)]))
    if only is None:  # the first missing-item scenario whose order its day names alone is told by its day
        for i, sc in enumerate(out):
            if sc.case_type != "item_missing_or_wrong":
                continue
            newest = conn.execute("select order_id, status, placed_at, delivered_at from orders.orders where "
                                  "customer_id = %s and placed_at <= %s order by placed_at desc limit %s",
                                  (sc.customer_id, now, RECENT_ORDERS)).fetchall()
            word = _by_day(sc.order_id, sc.case_type, newest, today)
            if word:
                out[i] = replace(sc, scenario_id=sc.scenario_id + "/by_day", names_order_by=word)
                break
    return out


def load_hard(conn: psycopg.Connection) -> list[Scenario]:
    """the hard set from the truth in this database; deterministic for one seed."""
    faults = conn.execute("select * from truth.order_faults order by fault_id").fetchall()
    out: list[Scenario] = []
    used: set[str] = set()
    for n, (fault_type, variant) in enumerate(HARD_NO_ITEM):
        fault = next((f for f in faults if (f["fault_type"], f["variant"]) == (fault_type, variant)), None)
        if fault is not None:
            used.add(fault["fault_id"])
            out.append(replace(_scenario(conn, fault, f"{fault_type}/{variant}/no_item", PERSONAS[n % 3]),
                               names_items=False))
    by_customer: dict[str, list[dict]] = {}
    for f in faults:
        if CASE_TYPE[f["fault_type"]] != "where_is_my_order" and f["fault_id"] not in used:
            by_customer.setdefault(f["customer_id"], []).append(f)
    pairs = [fs[:2] for _, fs in sorted(by_customer.items()) if len(fs) >= 2][:HARD_PAIRS]
    for n, (first, second) in enumerate(pairs):
        persona = PERSONAS[n % 3]
        one = _scenario(conn, first, f"{first['fault_type']}/{first['variant']}", persona)
        two = _scenario(conn, second, f"{second['fault_type']}/{second['variant']}", persona)
        out.append(replace(one, scenario_id=f"pair/{one.scenario_id}+{two.scenario_id}", also=two))
    return out


def _scenario(conn: psycopg.Connection, fault: dict, scenario_id: str, persona: str,
              names_order_by: str = "id") -> Scenario:
    name = conn.execute("select name from customer.customers where customer_id = %s",
                        (fault["customer_id"],)).fetchone()["name"]
    skus = fault["claim"].get("skus", [])
    items = [{"sku": r["sku"], "name": r["name"]} for r in conn.execute(
        "select sku, name from orders.order_lines where order_id = %s and sku = any(%s) order by line_no",
        (fault["order_id"], skus)).fetchall()]
    return Scenario(scenario_id=scenario_id, fault_id=fault["fault_id"], fault_type=fault["fault_type"],
                    variant=fault["variant"], customer_id=fault["customer_id"], customer_name=name,
                    order_id=fault["order_id"], case_type=CASE_TYPE[fault["fault_type"]],
                    summary=fault["claim"]["summary"], items=items, code=fault["claim"].get("code"),
                    names_order_by=names_order_by, persona=persona, owed_paise=fault["expected"]["owed_paise"],
                    proven_by_records=fault["expected"]["proven_by_records"], facts=fault["facts"],
                    asks_for=ASKS_FOR.get((fault["fault_type"], fault["variant"]), "check"))


def short_name(name: str) -> str:
    return name.split("(")[0].strip()


def opening(s: Scenario) -> str:
    """a fixed first message for the scenario (the b.6 runs, and the tests): the same words for every system. a
    second problem follows in the same message."""
    first = _one_problem(s)
    if s.also is None:
        return first
    second = _one_problem(s.also)
    return f"{first} Also, {second[0].lower()}{second[1:]}"


def _one_problem(s: Scenario) -> str:
    which = f"order {s.order_id}" if s.names_order_by == "id" else f"{s.names_order_by.replace('_', ' ')}'s order"
    items = " and ".join(short_name(i["name"]) for i in s.items)
    if s.case_type == "item_missing_or_wrong":
        ask = ASK_WORDS[s.asks_for]
        if not s.names_items:
            return f"Something did not come in my {which}. {ask}"
        return f"The {items} did not come in my {which}. {ask}"
    if s.case_type == "offers_coupons":
        return f"My coupon {s.code or ''} did not work properly on {which}.".replace("  ", " ")
    if s.fault_type == "duplicate_capture":
        return f"I was charged twice for {which}."
    return f"Money was taken for {which} but the payment failed."
