"""c22 action gateway, hivemind's side: the only way hivemind changes anything at the business.

every action carries a key (`<case>:remedy:1`, `<case>:cancel`, `<case>:outcome:<n>`). when the business cannot be
reached or does not answer in time, the same request goes again under the same key: if the first one went through,
the business hands back its first answer and nothing is done twice (blueprint s27). a refusal (the business's own
re-check says no) is not tried again; a person decides.

the decision record goes with the action: the business keeps it in the same database step as the money.
"""

import random
from typing import Callable

from hivemind_agents.backend_client import BackendClient, BackendError
from hivemind_agents.contracts import CaseRef, Decision, FactSheet, Issue

TRIES = 3
WAITS = (0.5, 1.5)  # seconds before the second and third tries


class StillPending(BackendError):
    """the business did not answer after every try: the case waits on the system."""


def remedy_key(case_id: str) -> str:
    return f"{case_id}:remedy:1"


def outcome_key(case_id: str, rng: random.Random) -> str:
    return f"{case_id}:outcome:{rng.randrange(1, 1_000_000)}"


def record(case: CaseRef, issue: Issue, sheet: FactSheet | None, decision: Decision) -> dict:
    """what was decided and on what (blueprint 11.7); the business adds the actions, the rules version and the
    hash chain."""
    body: dict = {
        "issue": {"type": case.case_type, "order_id": case.order_id, "items_mentioned": issue.items_mentioned,
                  "money_to": issue.money_to},
        "inputs": decision.inputs,
        "eligibility": {"rules_version": decision.eligibility.get("rules_version"),
                        "parts": [{k: p.get(k) for k in ("part", "outcome", "allowed", "max_amount_paise", "basis",
                                                          "requires", "rule", "why", "policy_refs")}
                                  for p in decision.eligibility.get("parts", [])]},
        "route": decision.route, "route_why": decision.why,
    }
    if sheet is not None:
        body["fact_sheet"] = {"round": sheet.round, "verdict": sheet.verdict, "source": sheet.source,
                              "claims": sheet.claims, "contradictions": len(sheet.contradictions),
                              "facts": len(sheet.facts)}
    if decision.proposal is not None:
        p = decision.proposal
        body["proposal"] = {"kind": p.kind, "destination": p.destination, "total_paise": p.total_paise,
                            "parts": [x.model_dump() for x in p.parts], "rationale_summary": p.rationale_summary,
                            "explanation_points": p.explanation_points, "source": p.source, "note": p.note}
    return body


def _again(call: Callable[[], dict], sleep: Callable[[float], None]) -> dict:
    for attempt in range(1, TRIES + 1):
        try:
            return call()
        except BackendError as exc:
            if attempt == TRIES:
                raise StillPending(str(exc)) from exc
            sleep(WAITS[attempt - 1])
    raise AssertionError("unreachable")


def act(case_id: str, decision: Decision, body: dict, backend: BackendClient,
        sleep: Callable[[float], None]) -> dict:
    """the refund, wallet credit or cancellation the decision proposes. raises ActionRefused or StillPending."""
    p = decision.proposal
    if p.kind == "cancel":
        request = {"case_id": case_id, "eligibility": decision.inputs, "decision": body}
        return _again(lambda: backend.cancel(request, f"{case_id}:cancel"), sleep)
    request = {"case_id": case_id, "parts": [{"part": x.part, "amount_paise": x.amount_paise} for x in p.parts],
               "eligibility": decision.inputs, "decision": body}
    pay = backend.refund if p.destination == "refund_to_source" else backend.wallet_credit
    return _again(lambda: pay(request, remedy_key(case_id)), sleep)


def keep_outcome(case_id: str, outcome: str, body: dict, backend: BackendClient, rng: random.Random,
                 sleep: Callable[[float], None]) -> dict:
    """a decision that moves no money, kept like any other. raises ActionRefused or StillPending."""
    key = outcome_key(case_id, rng)
    return _again(lambda: backend.record_outcome({"case_id": case_id, "outcome": outcome, "decision": body}, key),
                  sleep)
