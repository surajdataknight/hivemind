"""c18 policy & eligibility engine, hivemind's side: what the investigation concluded, put to the business's rules.

the rules live with the business and the action gateway runs them again before any money moves; this step only
says what was found, part by part, and reads back what the rules allow. no model.

a claimed line counts as corroborated only when code's own cross-checks point that way too: a model's
"corroborated" that no record supports is sent as uncorroborated. a payment part (an extra capture, a failed payment
that took money) is named from the payment check's findings; the business checks it against its own records.
"""

from hivemind_agents.backend_client import BackendClient
from hivemind_agents.capabilities import reconciler
from hivemind_agents.contracts import FactSheet, Finding, Issue

PAYMENT_FINDINGS = ("payment.duplicate_capture", "payment.failed_debit_not_reversed")


def risk_band(board: list[Finding]) -> str | None:
    """the risk check's band; none when it did not finish."""
    found = next((f for f in board if f.type == "risk.assessment"), None)
    return found.facts.get("band") if found else None


def line_claims(sheet: FactSheet, board: list[Finding]) -> list[dict]:
    """one claim per claimed order line, with the status the sheet gave it, held back by the cross-checks."""
    rows = reconciler.cross_checks(board)
    votes = reconciler._votes(board, rows)
    status_of = {c["sku"]: c["status"] for c in sheet.claims if c.get("sku")}
    claims = []
    for line in reconciler.claimed_lines(board):
        status = status_of.get(line["sku"], "uncorroborated")
        v = votes.get(line["sku"], {"supports": set(), "against": set()})
        if status == "corroborated" and (not v["supports"] or v["against"]):
            status = "uncorroborated"  # the model's word alone, with no record behind it
        claims.append({"part": f"line:{line['line_no']}", "status": status})
    return claims


def payment_claims(board: list[Finding]) -> list[dict]:
    claims = []
    for f in board:
        if f.type == "payment.duplicate_capture":
            claims += [{"part": f"capture:{c}", "status": "corroborated"} for c in f.facts["capture_ids"][1:]]
        elif f.type == "payment.failed_debit_not_reversed":
            claims.append({"part": f"intent:{f.facts['intent_id']}", "status": "corroborated"})
    return claims


def inputs_for(issue: Issue, sheet: FactSheet | None, board: list[Finding]) -> dict:
    """the request to the rules. a case with no investigation (a cancellation) sends no claims."""
    if sheet is None:
        return {"claims": []}
    claims = line_claims(sheet, board) if issue.case_type in ("item_missing_or_wrong", "quality") else []
    if issue.case_type in ("payment_failed", "refund"):
        claims = payment_claims(board)
    seen, unique = set(), []
    for c in claims:
        if c["part"] not in seen:
            seen.add(c["part"])
            unique.append(c)
    return {"claims": unique[:20], "risk_band": risk_band(board), "records_conflict": bool(sheet.contradictions),
            "verdict": sheet.verdict}


def evaluate(case_id: str, inputs: dict, backend: BackendClient) -> dict:
    return backend.eligibility(case_id, inputs)
