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
        self.reconciles: list = []  # evidence reconciler answers; when empty, a plain one (see plain_sheet)
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
            answer = self.reconciles.pop(0) if self.reconciles else plain_sheet(prompt)
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
    """the model calls a case makes besides the intake, the case lead and the reply: the notes read and the
    reconciler's sheets."""
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


def issue(case_type, order_id=None, coupon_code=None, items=(), amount=None, latest=False, day=None) -> dict:
    return {"case_type": case_type, "order_id": order_id, "items_mentioned": list(items),
            "claimed_amount_rupees": amount, "coupon_code": coupon_code, "about_latest_order": latest,
            "order_day": day}
