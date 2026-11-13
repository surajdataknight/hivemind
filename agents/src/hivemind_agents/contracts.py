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
    order_day: str | None = Field(default=None, description="the day the customer named the order by, with no order "
                                                            "id: today, yesterday, day_before_yesterday, a weekday, "
                                                            "or a date (YYYY-MM-DD); triage picks the order")


class IntakeV1(BaseModel):
    scope: Scope
    issues: list[Issue] = []
    sentiment: Literal["negative", "neutral", "positive"] = "neutral"
    urgency: Literal["low", "normal", "high"] = "normal"
    language: str = "en"
    needs_clarification: bool = False
    source: Literal["button", "rule", "llm"] = Field(description="what read the message: a button, a rule, the model")
    ask_again: bool = Field(default=False, description="the message does not answer our question: it is asked once "
                                                       "more, and nothing else is read from the message")
    still_waiting: bool = Field(default=False, description="our earlier question went unanswered twice and this "
                                                           "message gets a fixed reply: the question follows it")


class IntentHint(BaseModel):
    """what a button in the chat screen already knows, so no model has to read it."""

    case_type: CaseType
    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{6}$")
    coupon_code: str | None = Field(default=None, pattern=r"^[A-Z0-9]{3,20}$")
    items: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        default_factory=list, max_length=10, description="the order lines a 'which item?' button names")


class Turn(BaseModel):
    role: Literal["customer", "assistant"]
    text: str


class OrderRef(BaseModel):
    order_id: str
    status: str
    placed_at: datetime
    delivered_at: datetime | None = None


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
    facts: dict[str, Any] = Field(default_factory=dict, description="the values the claim rests on, as the read "
                                                                    "returned them (never a model's words)")


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


# c5, c6, c7: the plan, its checks, and what each check did


class TaskParams(BaseModel):
    """the only params a check can be given. fixed fields, not a free map: gemini's answer schema refuses a map."""

    order_id: str | None = None
    code: str | None = None


class TaskSpec(BaseModel):
    """one check as the case lead writes it. the validator decides whether it runs; the model never sets an order
    id or a code of its own (they come from the issue)."""

    task_id: str = Field(pattern=r"^t\d{1,2}$")  # no examples=: gemini's answer schema refuses the key
    capability: str = Field(max_length=40)
    params: TaskParams = Field(default_factory=TaskParams)
    depends_on: list[str] = Field(default_factory=list, max_length=12)
    reason: str = Field(default="", max_length=200)
    expects: str = Field(default="", max_length=200, description="what the check should show if the claim is true")


class PlanV1(BaseModel):
    """what the case lead answers: checks to add to the minimum plan, and when it would stop."""

    tasks: list[TaskSpec] = Field(default_factory=list, max_length=24)
    stop_condition: str = Field(default="", max_length=300)
    rationale_summary: str = Field(default="", max_length=500)


class Task(BaseModel):
    """a check the validator let through. it runs once, after the checks it depends on (blueprint 2.3, the task
    envelope: who asked for it and how deep in a chain of requests it sits)."""

    task_id: str
    capability: str
    params: dict[str, str]
    depends_on: list[str]
    mandatory: bool
    reason: str = ""
    expects: str = ""
    requested_by: str = "case_lead"  # minimum, case_lead, replan, or the id of the check that asked for it
    depth: int = 1                   # 1 for the plan's checks; a check asked for by another is one deeper

    @property
    def dedupe_key(self) -> str:
        return self.capability + ":" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))


class InfoRequest(BaseModel):
    """a check asking for another check (blueprint 2.3, info_requests). it never starts one itself: the validator
    decides, and the params come from the issue, as for every check."""

    capability: str = Field(max_length=40)
    reason: str = Field(default="", max_length=200)


class CheckResult(BaseModel):
    """what a check returns: its findings, and any other checks it asks for."""

    findings: list[Finding] = []
    info_requests: list[InfoRequest] = []


class Plan(BaseModel):
    """the validated plan for one case: the checks that run, the ones refused and why, and the minimum checks that
    cannot run (then the case does not run a plan at all)."""

    source: Literal["planner", "minimum"] = Field(description="planner: the case lead's plan was used; minimum: only the "
                                                              "minimum checks")
    tasks: list[Task] = []
    dropped: list[tuple[str, str]] = Field(default_factory=list, description="(task or capability, why)")
    missing: list[str] = Field(default_factory=list, description="minimum checks that cannot run")
    stop_condition: str = ""
    note: str = Field(default="", description="why the case lead's plan was not used, or that it came from the "
                                              "fallback model")
    replans: int = Field(default=0, description="how many times the case lead revised the plan")

    @property
    def runnable(self) -> bool:
        return not self.missing


class TaskOutcome(BaseModel):
    """what became of one check: done (with its findings), error, timeout (no answer within its time), blocked (a
    check it needs did not finish), not_run (the case ran out of its budget) or cancelled (the stop condition was met
    first; never a minimum check). none is dropped silently."""

    task_id: str
    status: Literal["done", "error", "timeout", "blocked", "not_run", "cancelled"]
    findings: list[Finding] = []
    info_requests: list[InfoRequest] = []
    error: str | None = None
    attempts: int = 0
    posted: bool = False


# c17


class FactSheet(BaseModel):
    """what the evidence reconciler makes of the case board (blueprint c17): facts each citing findings and the
    evidence behind them, the status of each part of the claim, contradictions, gaps, the follow-ups it asked for,
    and its verdict. the deciding steps (phase 3) read only this."""

    case_id: str
    round: int = Field(ge=1, description="1 after the first checks, 2 after the follow-ups")
    verdict: Literal["sufficient", "sufficient_with_uncertainty", "insufficient_escalate"]
    source: Literal["reconciler", "code"] = Field(description="code: no model answered; a person decides")
    facts: list[dict[str, Any]] = []           # statement, finding_ids, evidence_refs
    claims: list[dict[str, Any]] = []          # about, sku, status, finding_ids
    contradictions: list[dict[str, Any]] = []  # finding_ids (two), why
    gaps: list[str] = []
    follow_ups: list[str] = []
    cross_checks: list[dict[str, Any]] = []    # what code read from the findings together
    note: str = ""                             # why the verdict was changed, or why no model answered
    dropped: list[str] = []                    # what the model said that had no proof on the board


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


def add_outcomes(left: dict[str, TaskOutcome] | None, right: dict[str, TaskOutcome] | None) -> dict[str, TaskOutcome]:
    """one outcome per check, keyed by case and check id; a later write of the same key replaces the earlier one."""
    return {**(left or {}), **(right or {})}


class RunState(TypedDict, total=False):
    """the state of one run of the case graph: one customer message in, one reply out. a message can hold several
    issues; each gets its own case and adds its piece to the reply."""

    conversation_id: str
    identity: Identity
    raw_text: str
    hint: IntentHint | None
    history: list[Turn]
    pending: dict[str, Any] | None
    pending_chips: list[dict]         # the buttons that came with our open question, shown again when it is re-asked
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
    plan: Plan | None                 # the plan of the case being worked (c5, c6)
    plan_started: float | None        # when the plan started, on the monotonic clock: the case's time budget
    plan_usage: tuple[int, int] | None  # model calls and tokens used before the plan: its model budget starts there
    follow_ups: list[str] | None      # what the evidence reconciler still needs once the checks are done (c5 revises)
    fact_sheet: FactSheet | None      # the reconciler's latest sheet for the case being worked (c17)
    # what the run adds up
    pieces: Annotated[list[Reply], add_pieces]
    boards: Annotated[dict[str, list[Finding]], add_to_boards]
    outcomes: Annotated[dict[str, TaskOutcome], add_outcomes]   # keys: "<case id>/<check id>" (c7)
    reply: Reply
