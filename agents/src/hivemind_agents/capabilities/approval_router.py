"""c21 approval router: whether support acts alone, a person decides, or nothing is owed. code only.

the whole case goes to a person when any part of it needs one. until the independent reviewer exists (phase 4),
a claim refund it would review waits for a person instead: over the case limit, a risk band that is not low (or
unknown), records that contradict each other, or a message with several issues. a payment error the business's own
records show (charged twice, a failed payment that took money) is the business's mistake and is always put right.
money proven owed and benefit of doubt (wallet only) in one case are paid alone only when the customer
asked for the wallet: otherwise the owed money would go to the wallet unasked, and a person decides.
a note from a delivery partner or the store that the quarantined reader labelled as giving instructions sends the
whole case to a person, whatever the rules allow (blueprint t1, t2): someone tried to steer this case, so nothing is
paid on it alone, benefit of doubt least of all. the note's words still reach nothing; only its id is named.
"""

from hivemind_agents.contracts import FactSheet, Finding

CLAIM_BASES = ("proven", "benefit_of_doubt")


def route(answer: dict, sheet: FactSheet | None, several_issues: bool, money_to: str | None = None,
          board: list[Finding] | None = None) -> tuple[str, list[str]]:
    """(auto | person | nothing_owed | already_refunded, why)."""
    parts = answer["parts"]
    person = steering_notes(board or []) + [f"{p['part']}: {p['why']}" for p in parts if p["outcome"] == "person"]
    if person:
        return "person", person
    remedies = [p for p in parts if p["outcome"] == "remedy"]
    if not remedies:
        if parts and all(p["rule"] == "already_refunded" for p in parts):
            return "already_refunded", [p["why"] for p in parts]
        return "nothing_owed", [p["why"] for p in parts] or ["no part to decide"]
    approval = [f"{p['part']}: {p['why']}" for p in remedies if p["requires"] != "auto"]
    if approval:
        return "person", approval
    why = []
    claim = [p for p in remedies if p["basis"] in CLAIM_BASES]
    if claim:
        total = sum(p["max_amount_paise"] for p in claim)
        if total > answer["limits"]["auto_limit_per_case_paise"]:
            why.append("review: over the case limit")
        if answer.get("risk_band") != "low":
            why.append(f"review: risk band {answer.get('risk_band') or 'unknown'}")
        if sheet is not None and sheet.contradictions:
            why.append("review: the records contradict each other")
        if several_issues:
            why.append("review: several issues in one message")
    if why:
        return "person", why
    bases = {p["basis"] for p in remedies}
    if "benefit_of_doubt" in bases and bases - {"benefit_of_doubt"} and money_to != "wallet":
        return "person", ["owed money and benefit of doubt in one case, and the customer did not ask for the wallet"]
    if remedies[0]["kind"] == "refund" and not ways(remedies):
        return "person", ["no one way to pay every part"]
    return "auto", ["within the rules and the limits"]


def ways(remedies: list[dict]) -> list[str]:
    """the ways of paying allowed for every part, in the rules' order."""
    allowed = [w for w in remedies[0]["allowed"] if all(w in p["allowed"] for p in remedies)]
    return allowed


def steering_notes(board: list[Finding]) -> list[str]:
    """one reason per note on the case's board labelled as giving instructions, named by its id only."""
    why: list[str] = []
    for f in board:
        if f.type == "delivery.partner_note" and f.facts.get("label") == "note_gives_instructions":
            reason = f"a delivery partner's note gives instructions ({f.facts.get('note_id')})"
        elif f.type == "store.incident" and f.facts.get("note_label") == "note_gives_instructions":
            reason = f"a store incident note gives instructions ({f.facts.get('incident_id')})"
        else:
            continue
        if reason not in why:
            why.append(reason)
    return why
