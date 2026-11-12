"""the typed contracts between capabilities. nothing passes from one capability to the next except these."""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field

CaseType = Literal["where_is_my_order", "item_missing_or_wrong", "quality", "cancellation", "refund",
                   "payment_failed", "delivery_partner_behaviour", "offers_coupons"]
Scope = Literal["in_scope", "out_of_scope", "safety", "legal", "small_talk"]


# c1


class Identity(BaseModel):
    """who the chat belongs to. the case token itself is not in here; the backend client holds it."""

    customer_id: str
    session_id: str
    scopes: list[str]


# c2


class GuardResult(BaseModel):
    sanitised_text: str = Field(description="the message with personal data replaced by tokens such as <PHONE_1>")
    flags: list[Literal["contains_payment_credential"]] = []
    pii_found: dict[str, int] = Field(default_factory=dict, description="how many of each kind were replaced")


# c3


class Issue(BaseModel):
    case_type: CaseType
    order_id: str | None = None
    items_mentioned: list[str] = []
    claimed_amount_paise: int | None = None
    coupon_code: str | None = None
    latest_order: bool = Field(default=False, description="the customer said their last order, with no order id")


class IntakeV1(BaseModel):
    scope: Scope
    issues: list[Issue] = []
    sentiment: Literal["negative", "neutral", "positive"] = "neutral"
    urgency: Literal["low", "normal", "high"] = "normal"
    language: str = "en"
    needs_clarification: bool = False
    source: Literal["button", "rule", "llm"] = Field(description="what read the message: a button, a rule, the model")


class IntentHint(BaseModel):
    """what a button in the chat screen already knows, so no model has to read it."""

    case_type: CaseType
    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{6}$")
    coupon_code: str | None = Field(default=None, pattern=r"^[A-Z0-9]{3,20}$")


class Turn(BaseModel):
    role: Literal["customer", "assistant"]
    text: str


class OrderRef(BaseModel):
    order_id: str
    status: str
    placed_at: datetime


# c4


class CaseRef(BaseModel):
    case_id: str
    case_type: CaseType
    order_id: str | None
    state: str
    opened: bool


class Lease(BaseModel):
    """this run's hold on a case: one run per case. a write carries the fencing number; once another run has taken
    the case over, the number is old and the write is refused."""

    case_id: str
    holder: str
    fence: int


# the case board


class Finding(BaseModel):
    """one fact an investigator found, tied to the reads that prove it (blueprint 2.3, the result envelope)."""

    finding_id: str = Field(default_factory=lambda: "f_" + uuid4().hex[:16])
    case_id: str
    task_id: str
    capability: str
    type: str = Field(pattern=r"^[a-z_]+\.[a-z_]+$", examples=["delivery.pod_mismatch"])
    claim: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1, description="evidence ids of reads made in this case")
    confidence: Literal["low", "medium", "high"]
    confidence_basis: str | None = None


# c12


class PromotionFinding(BaseModel):
    """what the promotion engine says about one coupon on one order."""

    order_id: str
    code: str
    outcome: Literal["unknown_code", "not_eligible", "not_entered", "applied", "applied_too_little",
                     "wrongly_rejected"]
    failed_rule: str | None = None
    item_total_paise: int
    min_cart_paise: int | None = None
    due_discount_paise: int
    applied_discount_paise: int
    difference_paise: int
    offer_kind: str | None = None
    offer_value: int | None = None
    max_discount_paise: int | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    placed_at: datetime
    checkout_status: str | None = None
    checkout_reason: str | None = None
    evidence_refs: list[str]

    @property
    def owed(self) -> bool:
        return self.outcome in ("applied_too_little", "wrongly_rejected")


# c16


class PolicySection(BaseModel):
    policy_id: str
    version: str
    title: str
    text: str
    evidence_ref: str

    @property
    def ref(self) -> str:
        return f"{self.policy_id}@{self.version}"


# replies


class Chip(BaseModel):
    """a button under a reply; pressing it sends `text` with this intent."""

    label: str
    text: str
    intent: IntentHint


class Reply(BaseModel):
    kind: str = Field(description="which reply this is, e.g. order_status, coupon_explained, out_of_scope")
    text: str = Field(description="as stored: tokens stand in for personal data")
    chips: list[Chip] = []
    case_id: str | None = None
    case_state: str | None = None
    llm_calls: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


def add_to_boards(left: dict[str, list[Finding]] | None,
                  right: dict[str, list[Finding]] | None) -> dict[str, list[Finding]]:
    """the case board's reducer: findings are only ever added, one list per case, each finding once
    (blueprint b.5: a typed state channel with an append-only reducer)."""
    boards = {case_id: list(findings) for case_id, findings in (left or {}).items()}
    for case_id, findings in (right or {}).items():
        board = boards.setdefault(case_id, [])
        known = {f.finding_id for f in board}
        board.extend(f for f in findings if f.finding_id not in known)
    return boards


def add_pieces(left: list[Reply] | None, right: list[Reply] | None) -> list[Reply]:
    return (left or []) + (right or [])


class RunState(TypedDict, total=False):
    """the state of one run of the case graph: one customer message in, one reply out. a message can hold several
    issues; each gets its own case and adds its piece to the reply."""

    conversation_id: str
    identity: Identity
    raw_text: str
    hint: IntentHint | None
    history: list[Turn]
    pending: dict[str, Any] | None
    guard: GuardResult
    recent_orders: list[OrderRef]
    intake: IntakeV1
    todo: list[Issue]                 # issues not worked yet, in the order the customer wrote them
    seen: list[str]                   # case type and order of each issue worked in this run
    ask: Reply | None                 # the one question this reply ends with; later issues that need one wait in it
    # the issue being worked, cleared before the next one
    issue: Issue | None
    case: CaseRef | None
    finding: PromotionFinding | None
    policy: list[PolicySection] | None
    piece: Reply | None
    # what the run adds up
    pieces: Annotated[list[Reply], add_pieces]
    boards: Annotated[dict[str, list[Finding]], add_to_boards]
    reply: Reply
