"""a scripted model for the tests, and the intake answers a correct model would give."""

import json
import re

from hivemind_agents.capabilities import quarantined_reader, reconciler
from hivemind_agents.llm import Completion


class ScriptedModel:
    """stands in for a provider. intake answers are queued by the test; the reply, unless a test queues one,
    repeats the finding the facts sheet gives, which is what a faithful model would do."""

    def __init__(self):
        self.intakes: list = []
        self.replies: list = []
        self.labels: list = []  # quarantined reader answers; when empty, a faithful reading of the note
        self.reconciles: list = []  # evidence reconciler answers (or a function of the prompt); when empty, a
        #                             plain one (see plain_sheet)
        self.calls: list[dict] = []
        self.down = False

    def generate(self, model, system, prompt, temperature, schema, timeout_s=None):
        self.calls.append({"kind": "generate", "model": model, "system": system, "prompt": prompt})
        if self.down:
            raise RuntimeError("429 quota used up")
        if system == quarantined_reader.SYSTEM:  # notes are read side by side: answered by note, not from the queue
            self.calls[-1]["kind"] = "label"
            label = self.labels.pop(0) if self.labels else faithful_label(prompt)
            return Completion(json.dumps({"label": label}), 120, 8)
        if system == reconciler.SYSTEM:
            self.calls[-1]["kind"] = "reconcile"
            answer = self.reconciles.pop(0) if self.reconciles else plain_sheet
            answer = answer(prompt) if callable(answer) else answer
            return Completion(answer if isinstance(answer, str) else json.dumps(answer), 900, 150)
        if not self.intakes:
            raise AssertionError(f"the model was asked to read a message the test did not expect:\n{prompt}")
        answer = self.intakes.pop(0)
        return Completion(answer if isinstance(answer, str) else json.dumps(answer), 400, 60)

    def stream(self, model, system, prompt, temperature, on_done, timeout_s=None):
        self.calls.append({"kind": "stream", "model": model, "system": system, "prompt": prompt})
        if self.down:
            raise RuntimeError("429 quota used up")
        text = self.replies.pop(0) if self.replies else faithful_reply(prompt)
        if isinstance(text, Exception):
            raise text
        for start in range(0, len(text), 9):
            yield text[start:start + 9]
        on_done(Completion(text, 500, 80))


LABEL_WORDS = [  # the first that fits wins
    ("note_gives_instructions", r"ignore|refund|approve|assistant|system:|mark all"),
    ("note_mentions_handover_to_someone_else", r"family|mother|security|guard|neighbour"),
    ("note_mentions_out_of_stock", r"bin empty|out of stock|not on (the )?shelf"),
    ("note_mentions_missing_item", r"missing|not handed"),
    ("note_mentions_damage", r"damage|torn|spill|spoil|expired"),
    ("note_mentions_delay", r"absent|queue|power|slow"),
    ("note_mentions_customer_unavailable", r"not reachable|unavailable|no answer"),
]


def faithful_label(prompt: str) -> str:
    """the label a careful reader gives the quoted note, from the labels the prompt offers."""
    note = re.search(r'"""\n(.*)\n"""', prompt, re.S).group(1).lower()
    offered = set(re.findall(r"^- (note_[a-z_]+|none):", prompt, re.M))
    for label, pattern in LABEL_WORDS:
        if label in offered and re.search(pattern, note):
            return label
    return "none"


# the case lead's answer that adds no check to the minimum plan
NO_EXTRA_CHECKS = {"tasks": [], "stop_condition": "", "rationale_summary": "the minimum checks"}


def label_calls(model: "ScriptedModel") -> int:
    return sum(1 for c in model.calls if c["kind"] == "label")


def side_calls(model: "ScriptedModel") -> int:
    """the model calls a case makes besides the intake, the case lead and the reply: the notes read, the
    reconciler's sheets and the resolution planner's choices."""
    return sum(1 for c in model.calls if c["kind"] in ("label", "reconcile"))


def calls_of(model: "ScriptedModel", kind: str) -> list[dict]:
    return [c for c in model.calls if c["kind"] == kind]


def plain_sheet(prompt: str) -> dict:
    """a reconciler answer that only restates each finding, settles no part of the claim, asks for nothing more,
    and so leaves the case uncertain: a person decides."""
    labels = re.findall(r"^(F\d+) \[", prompt, re.M)
    return {"facts": [{"statement": f"finding {label} as posted", "cites": [label]} for label in labels],
            "claims": [], "contradictions": [], "gaps": [], "follow_ups": [],
            "verdict": "sufficient_with_uncertainty"}


def careful_sheet(prompt: str) -> dict:
    """a reconciler answer that reads the code's cross-checks: a claimed line is corroborated when a record points
    to it and none against it, contradicted when only records against it, else unsettled; c1 (a payment) is
    corroborated when the payment records show it. contradictions are the cross-checks' conflicts."""
    labels = re.findall(r"^(F\d+) \[", prompt, re.M)
    lines = re.findall(r"^- (L\d+): .* \(([^()]+)\), \d+ x", prompt, re.M)
    rows = re.findall(r"^- (.+?): ([a-z_]+) \(([a-z]+)\), from ([F\d, ]*):", prompt, re.M)
    claims = []
    for label, sku in lines:
        mine = [r for r in rows if r[0] in (sku, "the bags")]
        up = [r for r in mine if r[2] == "supports"]
        down = [r for r in mine if r[2] == "against"]
        status = "corroborated" if up and not down else "contradicted" if down and not up else "uncorroborated"
        cites = sorted({c.strip() for r in (up + down) for c in r[3].split(",") if c.strip()})
        claims.append({"about": label, "status": status, "cites": cites})
    if re.search(r"^- C1: ", prompt, re.M):
        paid = [r for r in rows if r[0] == "the payment" and r[2] == "supports"]
        claims.append({"about": "C1", "status": "corroborated" if paid else "uncorroborated",
                       "cites": sorted({c.strip() for r in paid for c in r[3].split(",") if c.strip()})})
    conflicts = [{"first": r[3].split(",")[0].strip(), "second": r[3].split(",")[1].strip(), "why": r[1]}
                 for r in rows if r[2] == "conflict" and "," in r[3]]
    settled = all(c["status"] != "uncorroborated" for c in claims)
    return {"facts": [{"statement": f"finding {label} as posted", "cites": [label]} for label in labels],
            "claims": claims, "contradictions": conflicts, "gaps": [], "follow_ups": [],
            "verdict": "sufficient" if settled and claims and not conflicts else "sufficient_with_uncertainty"}


def faithful_reply(prompt: str) -> str:
    found = re.search(r"^what the check found: (.*)$", prompt, re.M).group(1)
    if "you cannot correct the amount yourself" in prompt:
        return (f"{found} I can't correct this myself, so I've recorded it for a person in customer care to "
                "review. You'll get the update here in this chat.")
    return f"Thanks for checking with me. {found}"


def draft(scope="in_scope", issues=(), language="en", sentiment="neutral", urgency="normal", unclear=False) -> dict:
    """what a model that reads the message correctly returns for the intake form."""
    return {"scope": scope, "issues": list(issues), "sentiment": sentiment, "urgency": urgency,
            "language": language, "needs_clarification": unclear}


def issue(case_type, order_id=None, coupon_code=None, items=(), amount=None, latest=False, day=None,
          money_to="not_said") -> dict:
    return {"case_type": case_type, "order_id": order_id, "items_mentioned": list(items),
            "claimed_amount_rupees": amount, "coupon_code": coupon_code, "about_latest_order": latest,
            "order_day": day, "money_to": money_to}
