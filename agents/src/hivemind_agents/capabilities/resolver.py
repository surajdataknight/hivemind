"""c19 resolution planner: how the remedies the rules allow are paid, and the points the reply explains.

the amounts are the rules' (each part at the most it may be). the way is chosen in code from the ways the rules
allow for every part and what the customer asked for, which intake read: the wallet when they asked for it, a
refund to the original payment method when they asked for that or did not say (a wallet credit they did not want
keeps their money in the app). the reason in the record and the points the reply explains are written here from
the facts, so neither can say something the customer or the records did not. no model is asked: with the way read
by intake and the amounts fixed by the rules, nothing is left to judge (seen 2026-11-13: a model chose the card and
recorded "the customer asked for their money back" for a customer who wrote only "please check").
"""

from hivemind_agents.capabilities import approval_router
from hivemind_agents.contracts import Proposal, ProposedPart

MAX_POINTS = 3
SOURCE, WALLET = "refund_to_source", "wallet_credit"

BASIS_WORDS = {
    "proven": "the records show it was not delivered",
    "benefit_of_doubt": "the records could not confirm it either way; it is credited as a goodwill gesture",
    "payment_error": "the payment records show money was taken that should not have been",
    "order_cancelled": "the order is cancelled before packing; what was paid goes back",
}
ASKED_WORDS = {"card_or_bank": "the original payment method", "wallet": "the HiveMart wallet"}
WAY_WORDS = {SOURCE: "the original payment method", WALLET: "the HiveMart wallet"}


def parts_of(remedies: list[dict]) -> list[ProposedPart]:
    return [ProposedPart(part=p["part"], name=p.get("name"), amount_paise=p["max_amount_paise"], basis=p["basis"])
            for p in remedies if p["max_amount_paise"] > 0]


def code_points(parts: list[ProposedPart]) -> list[str]:
    return [f"{p.name or 'The amount'}: {BASIS_WORDS.get(p.basis, p.basis)}." for p in parts][:MAX_POINTS]


def choose(ways: list[str], money_to: str | None) -> tuple[str, str, str]:
    """(way, source, reason for the record)."""
    asked = {"card_or_bank": SOURCE, "wallet": WALLET}.get(money_to or "")
    if len(ways) == 1:
        reason = f"only {WAY_WORDS[ways[0]]} is allowed"
        if asked and asked != ways[0]:
            reason += f"; the customer asked for {ASKED_WORDS[money_to]}"
        return ways[0], "no_choice", reason
    if asked in ways:
        return asked, "asked", f"the customer asked for {ASKED_WORDS[money_to]}"
    way = SOURCE if SOURCE in ways else ways[0]
    return way, "default", f"the customer did not say where; {WAY_WORDS[way]}"


def propose(answer: dict, money_to: str | None) -> Proposal:
    remedies = [p for p in answer["parts"] if p["outcome"] == "remedy"]
    parts = parts_of(remedies)
    total = sum(p.amount_paise for p in parts)
    if remedies[0]["kind"] == "cancel":
        return Proposal(kind="cancel", destination=SOURCE, parts=parts, total_paise=total,
                        rationale_summary="cancelled before packing; what was paid goes back", source="no_choice")
    way, source, reason = choose(approval_router.ways(remedies), money_to)
    return Proposal(kind="refund", destination=way, parts=parts, total_paise=total, rationale_summary=reason,
                    explanation_points=code_points(parts), source=source)
