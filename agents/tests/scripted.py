"""a scripted model for the tests, and the intake answers a correct model would give."""

import json
import re

from hivemind_agents.llm import Completion


class ScriptedModel:
    """stands in for a provider. intake answers are queued by the test; the reply, unless a test queues one,
    repeats the finding the facts sheet gives, which is what a faithful model would do."""

    def __init__(self):
        self.intakes: list = []
        self.replies: list = []
        self.calls: list[dict] = []
        self.down = False

    def generate(self, model, system, prompt, temperature, schema):
        self.calls.append({"kind": "generate", "model": model, "system": system, "prompt": prompt})
        if self.down:
            raise RuntimeError("429 quota used up")
        if not self.intakes:
            raise AssertionError(f"the model was asked to read a message the test did not expect:\n{prompt}")
        answer = self.intakes.pop(0)
        return Completion(answer if isinstance(answer, str) else json.dumps(answer), 400, 60)

    def stream(self, model, system, prompt, temperature, on_done):
        self.calls.append({"kind": "stream", "model": model, "system": system, "prompt": prompt})
        if self.down:
            raise RuntimeError("429 quota used up")
        text = self.replies.pop(0) if self.replies else faithful_reply(prompt)
        if isinstance(text, Exception):
            raise text
        for start in range(0, len(text), 9):
            yield text[start:start + 9]
        on_done(Completion(text, 500, 80))


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


def issue(case_type, order_id=None, coupon_code=None, items=(), amount=None) -> dict:
    return {"case_type": case_type, "order_id": order_id, "items_mentioned": list(items),
            "claimed_amount_rupees": amount, "coupon_code": coupon_code}
