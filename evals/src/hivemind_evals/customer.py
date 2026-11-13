"""the simulated customer (record 014, L5): a model with a persona and a secret goal chats with hivemind the way a
person would. it knows only what the customer would know: their problem, their order, their own words; never the
truth's cause. the scripted customer says fixed words, for the tests and the b.6 runs.
"""

from typing import Protocol

from pydantic import BaseModel, Field

from hivemind_agents.llm import ModelRouter
from hivemind_evals.scenarios import Scenario, opening, short_name

MAX_TURNS = 6


class CustomerTurn(BaseModel):
    message: str = Field(default="", max_length=400, description="what the customer types next; empty when done")
    press: str | None = Field(default=None, max_length=120, description="the text of a button to press instead of "
                                                                       "typing, exactly as offered")
    done: bool = Field(description="true when hivemind has answered the problem or said a person will review it")


class Customer(Protocol):
    def next(self, transcript: list[dict]) -> CustomerTurn: ...


SYSTEM = (
    "You play a customer of HiveMart, a grocery delivery app, chatting with its support assistant. Stay in "
    "character. Write short chat messages, as a real customer types them on a phone. Never say you are an AI or "
    "that this is a test. Answer only in the JSON shape asked for."
)


def _problem(s: Scenario) -> str:
    if s.names_order_by == "id":
        which = f"your order {s.order_id}; write its id when you first say the problem"
    else:
        which = (f"the order you mean only as \"{s.names_order_by.replace('_', ' ')}'s order\" (never write its id "
                 "unless the assistant asks you to pick)")
    items = ", ".join(i["name"] for i in s.items) or "none"
    told = "" if s.names_items else (" In your first message do not name the items: say only that something was "
                                     "missing. Name them only when the assistant asks which item, by pressing its "
                                     "button or typing the names.")
    return (f"{s.summary}. Items: {items}." + (f" The coupon code: {s.code}." if s.code else "")
            + f" Refer to {which}.{told}")


def build_prompt(s: Scenario, transcript: list[dict]) -> str:
    lines = "\n".join(f"{t['role']}: {t['text']}" + (f" [buttons: {', '.join(t['buttons'])}]" if t.get("buttons")
                                                     else "") for t in transcript) or "(nothing yet: you start)"
    if s.also is None:
        problem = f"Your problem: {_problem(s)}\n"
        ending = "When the assistant has answered your problem"
    else:
        problem = (f"You have two problems; write both in your first message.\n1. {_problem(s)}\n"
                   f"2. {_problem(s.also)}\n")
        ending = "When the assistant has answered both problems"
    orders = " or ".join(p.order_id for p in s.parts())
    return (f"You are {s.customer_name}, {s.persona}.\n{problem}"
            f"If the assistant asks which order, press the button for your order ({orders}) or type its id. "
            "Answer its questions briefly and truthfully from the problem above; do not invent other problems. "
            f"{ending}, or says a person in customer care will review it, end the chat (done: true).\n\n"
            f"The chat so far:\n{lines}\n\nYour next turn: message, press (or null), done.")


class SimulatedCustomer:
    def __init__(self, scenario: Scenario, llm: ModelRouter):
        self.s, self.llm = scenario, llm

    def next(self, transcript: list[dict]) -> CustomerTurn:
        return self.llm.structured("simulated_customer", SYSTEM, build_prompt(self.s, transcript), CustomerTurn)


class ScriptedCustomer:
    """the fixed opening; when asked which order, the button for the scenario's order; when asked which item (the
    question names its order), the item's button, or the items' names typed when there are several; then done."""

    def __init__(self, scenario: Scenario):
        self.s = scenario

    def next(self, transcript: list[dict]) -> CustomerTurn:
        if not transcript:
            return CustomerTurn(message=opening(self.s), done=False)
        last = transcript[-1]
        if not last.get("asks"):
            return CustomerTurn(done=True)
        buttons = last.get("buttons", [])
        for part in self.s.parts():
            if part.order_id in buttons:
                return CustomerTurn(press=part.order_id, done=False)
        about = next((p for p in self.s.parts() if p.order_id in last["text"] and p.items), None)
        if about is not None:
            names = [i["name"] for i in about.items]
            if len(names) == 1 and names[0] in buttons:
                return CustomerTurn(press=names[0], done=False)
            return CustomerTurn(message=" and ".join(short_name(n) for n in names), done=False)
        return CustomerTurn(done=True)
