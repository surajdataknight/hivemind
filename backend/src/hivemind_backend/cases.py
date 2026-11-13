"""the support case's state machine: which moves are legal.

    main path:  OPEN ─► INVESTIGATING ─► DECIDING ─► ACTIONING ─► RESOLVED ─► CLOSED
    waits:      AWAITING_CUSTOMER, AWAITING_HUMAN, ESCALATED, PENDING_SYSTEM (a system was down; tried again later)
    after:      RESOLVED ─► REOPENED (the customer disputes the outcome)

the states are the blueprint's (deliverable 5, section 5.5); each move follows a step of the case graph (3.1):
investigating covers plan, run and reconcile; deciding covers eligibility, resolve, review and route; awaiting
human is an approval the rules ask for, escalated a case handed to a person with an open ticket. a person's
answer to either leads to acting, resolving, asking the customer, or deciding again on a person's finding (3.6).
a closed case never moves again; a resolved one can be reopened by the customer.
"""

STATES = ("OPEN", "INVESTIGATING", "DECIDING", "AWAITING_CUSTOMER", "AWAITING_HUMAN", "PENDING_SYSTEM", "ACTIONING",
          "ESCALATED", "RESOLVED", "CLOSED", "REOPENED")

# states a case can still be worked in; the others are done with
OPEN_STATES = tuple(s for s in STATES if s not in ("RESOLVED", "CLOSED"))

_A_PERSON_ANSWERED = {"ACTIONING", "RESOLVED", "AWAITING_CUSTOMER", "DECIDING"}

TRANSITIONS: dict[str, set[str]] = {
    # a fast path can act at once (cancel before packing) or a ticket can open at once (safety, legal)
    "OPEN": {"INVESTIGATING", "ACTIONING", "AWAITING_CUSTOMER", "ESCALATED"},
    # a short path answers and resolves here; a case with no path yet waits for a person
    "INVESTIGATING": {"DECIDING", "AWAITING_CUSTOMER", "AWAITING_HUMAN", "PENDING_SYSTEM", "ESCALATED", "RESOLVED"},
    "DECIDING": {"ACTIONING", "AWAITING_CUSTOMER", "AWAITING_HUMAN", "ESCALATED", "RESOLVED"},
    "AWAITING_CUSTOMER": {"INVESTIGATING", "DECIDING", "CLOSED"},
    "AWAITING_HUMAN": set(_A_PERSON_ANSWERED),
    "ESCALATED": set(_A_PERSON_ANSWERED),
    "PENDING_SYSTEM": {"INVESTIGATING", "DECIDING", "ACTIONING", "ESCALATED"},
    # the gateway refused the action on its own re-check: a person decides
    "ACTIONING": {"RESOLVED", "PENDING_SYSTEM", "AWAITING_HUMAN", "ESCALATED"},
    "RESOLVED": {"CLOSED", "REOPENED"},
    "CLOSED": set(),
    # blueprint 4.5: a second dispute always goes to a person
    "REOPENED": {"INVESTIGATING", "DECIDING", "AWAITING_HUMAN", "ESCALATED"},
}


def can_move(current: str, to: str) -> bool:
    return to in TRANSITIONS.get(current, set())
