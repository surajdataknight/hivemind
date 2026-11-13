"""the doorway to the business: hivemind reaches orders, delivery, promotions and cases only through these calls.

one client per run, bound to that run's case token. the customer is whoever the token names; no call here takes a
customer id.
"""

import copy
from typing import Any, Callable, Protocol

from hivemind_agents import evidence
from hivemind_agents.tracing import Tracer, trace_headers

# (case id, operation, body): keeps a read as proof for the case it was made for
Ledger = Callable[[str, str, dict], None]


class Http(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


class BackendError(Exception):
    """the backend could not answer (down, timeout, or an error that is not about the request itself)."""


class NotFound(Exception):
    pass


class ActionRefused(Exception):
    """the business said no to an action (a re-check failed, the key was used for another request, the case is not
    being decided): asking again would get the same answer."""

    def __init__(self, operation: str, status: int, detail: str):
        super().__init__(f"{operation}: http {status}: {detail}")
        self.status, self.detail = status, detail


class BackendClient:
    def __init__(self, http: Http, case_token: str, tracer: Tracer, ledger: Ledger | None = None):
        self._http = http
        self._auth = {"Authorization": f"Bearer {case_token}"}
        self._tracer = tracer
        self._ledger = ledger
        self.case_id: str | None = None  # the case being worked: every read is kept as proof for it
        self._closed = False

    def for_case(self, case_id: str) -> "BackendClient":
        """a client of its own for one check: its reads stay proof for this case, whatever the run works on next."""
        mine = copy.copy(self)
        mine.case_id, mine._closed = case_id, False
        return mine

    def close(self) -> None:
        """no further reads: a check given up on (timed out or cancelled) can no longer add proof to the case."""
        self._closed = True

    def _call(self, operation: str, method: str, path: str, *, params: dict | None = None,
              body: dict | None = None, key: str | None = None) -> dict:
        if self._closed:
            raise BackendError(f"{operation}: the check was stopped")
        headers = {**self._auth, **trace_headers()}
        if key:
            headers["Idempotency-Key"] = key
        if self.case_id:
            headers["X-Case-Id"] = self.case_id
        with self._tracer.tool(operation) as span:
            span.set_attribute("http.request.method", method)
            span.set_attribute("url.path", path)
            try:
                response = self._http.request(method, path, params=params, json=body, headers=headers)
            except Exception as exc:
                raise BackendError(f"{operation}: {type(exc).__name__}") from exc
            span.set_attribute("http.response.status_code", response.status_code)
            if key and 400 <= response.status_code < 500:  # an action the business refused, not a failure
                detail = _detail(response)
                span.set_attribute("hivemind.action.refused", detail[:200])
                raise ActionRefused(operation, response.status_code, detail)
            if response.status_code >= 400 and response.status_code != 404:
                raise BackendError(f"{operation}: http {response.status_code}")
            data = response.json()
            if "evidence_id" in data:
                span.set_attribute("hivemind.evidence_id", data["evidence_id"])
                if not evidence.matches(data):
                    raise BackendError(f"{operation}: the evidence id does not match the body")
                if self.case_id and self._ledger and not self._closed:
                    self._ledger(self.case_id, operation, data)
        if response.status_code == 404:  # an answer, not a failure: raised outside the span
            raise NotFound(operation)
        return data

    # identity
    def me(self) -> dict:
        return self._call("getMe", "GET", "/v1/me")

    def clock(self) -> dict:
        return self._call("getClock", "GET", "/v1/clock")

    # orders and delivery
    def my_orders(self, limit: int = 5) -> dict:
        return self._call("listMyOrders", "GET", "/v1/me/orders", params={"limit": limit})

    def order(self, order_id: str) -> dict:
        return self._call("getOrder", "GET", f"/v1/orders/{order_id}")

    def trip(self, order_id: str) -> dict | None:
        try:
            return self._call("getOrderTrip", "GET", f"/v1/orders/{order_id}/trip")
        except NotFound:
            return None  # no delivery partner assigned yet

    def proof_of_delivery(self, order_id: str) -> dict | None:
        try:
            return self._call("getProofOfDelivery", "GET", f"/v1/orders/{order_id}/pod")
        except NotFound:
            return None  # not handed over yet

    def partner_notes(self, order_id: str) -> dict:
        return self._call("listPartnerNotes", "GET", f"/v1/orders/{order_id}/partner-notes")

    # store and inventory
    def scans(self, order_id: str) -> dict:
        return self._call("getOrderScans", "GET", f"/v1/orders/{order_id}/scans")

    def store_incidents(self, store_id: str, since: str, until: str) -> dict:
        return self._call("listStoreIncidents", "GET", f"/v1/stores/{store_id}/incidents",
                          params={"since": since, "until": until})

    def stock_events(self, store_id: str, sku: str, since: str, until: str) -> dict:
        return self._call("listStockEvents", "GET", f"/v1/stores/{store_id}/stock-events",
                          params={"sku": sku, "since": since, "until": until})

    # payments and refunds
    def payments(self, order_id: str) -> dict:
        return self._call("listOrderPayments", "GET", f"/v1/orders/{order_id}/payments")

    def refunds(self, order_id: str) -> dict:
        return self._call("listOrderRefunds", "GET", f"/v1/orders/{order_id}/refunds")

    # the customer's support profile and risk signals: counts only
    def support_profile(self, window_days: int = 180) -> dict:
        return self._call("getSupportProfile", "GET", "/v1/me/support-profile", params={"window_days": window_days})

    def risk_signals(self, window_days: int = 180) -> dict:
        return self._call("getRiskSignals", "GET", "/v1/me/risk-signals", params={"window_days": window_days})

    # promotions
    def my_redemptions(self, limit: int = 20) -> dict:
        return self._call("listMyRedemptions", "GET", "/v1/me/redemptions", params={"limit": limit})

    def evaluate_promotion(self, order_id: str, code: str) -> dict:
        return self._call("evaluatePromotion", "GET", f"/v1/orders/{order_id}/promotion-evaluation",
                          params={"code": code})

    def offer(self, code: str) -> dict:
        return self._call("getOffer", "GET", f"/v1/promotions/offers/{code}")

    # policy
    def policy_sections(self, issue_type: str) -> dict:
        return self._call("listPolicySections", "GET", "/v1/policy/sections", params={"issue_type": issue_type})

    # cases
    def open_case(self, case_type: str, order_id: str | None) -> dict:
        return self._call("openCase", "POST", "/v1/cases", body={"issue_type": case_type, "order_id": order_id})

    def move_case(self, case_id: str, state: str, reason: str | None = None) -> dict:
        return self._call("moveCase", "POST", f"/v1/cases/{case_id}/state", body={"state": state, "reason": reason})

    # what the business's rules allow on a case (c18), and its actions (c22): each action carries a key, so asking
    # again with the same key never does it twice
    def eligibility(self, case_id: str, inputs: dict) -> dict:
        return self._call("evaluateEligibility", "POST", f"/v1/cases/{case_id}/eligibility", body=inputs)

    def refund(self, body: dict, key: str) -> dict:
        return self._call("refundToSource", "POST", "/v1/actions/refunds", body=body, key=key)

    def wallet_credit(self, body: dict, key: str) -> dict:
        return self._call("creditWallet", "POST", "/v1/actions/wallet-credits", body=body, key=key)

    def cancel(self, body: dict, key: str) -> dict:
        return self._call("cancelOrder", "POST", "/v1/actions/cancellations", body=body, key=key)

    def record_outcome(self, body: dict, key: str) -> dict:
        return self._call("recordOutcome", "POST", "/v1/actions/outcomes", body=body, key=key)


def _detail(response: Any) -> str:
    try:
        return str(response.json().get("detail", ""))
    except Exception:  # noqa: BLE001
        return ""
