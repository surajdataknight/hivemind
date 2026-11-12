"""the doorway to the business: hivemind reaches orders, delivery, promotions and cases only through these calls.

one client per run, bound to that run's case token. the customer is whoever the token names; no call here takes a
customer id.
"""

from typing import Any, Protocol

from hivemind_agents.tracing import Tracer, trace_headers


class Http(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


class BackendError(Exception):
    """the backend could not answer (down, timeout, or an error that is not about the request itself)."""


class NotFound(Exception):
    pass


class BackendClient:
    def __init__(self, http: Http, case_token: str, tracer: Tracer):
        self._http = http
        self._auth = {"Authorization": f"Bearer {case_token}"}
        self._tracer = tracer
        self.case_id: str | None = None

    def _call(self, operation: str, method: str, path: str, *, params: dict | None = None,
              body: dict | None = None) -> dict:
        headers = {**self._auth, **trace_headers()}
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
            if response.status_code >= 400 and response.status_code != 404:
                raise BackendError(f"{operation}: http {response.status_code}")
            data = response.json()
            if "evidence_id" in data:
                span.set_attribute("hivemind.evidence_id", data["evidence_id"])
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

    # promotions
    def my_redemptions(self, limit: int = 20) -> dict:
        return self._call("listMyRedemptions", "GET", "/v1/me/redemptions", params={"limit": limit})

    def evaluate_promotion(self, order_id: str, code: str) -> dict:
        return self._call("evaluatePromotion", "GET", f"/v1/orders/{order_id}/promotion-evaluation",
                          params={"code": code})

    def offer(self, code: str) -> dict:
        return self._call("getOffer", "GET", f"/v1/promotions/offers/{code}")

    # cases
    def open_case(self, case_type: str, order_id: str | None) -> dict:
        return self._call("openCase", "POST", "/v1/cases", body={"issue_type": case_type, "order_id": order_id})

    def move_case(self, case_id: str, state: str) -> dict:
        return self._call("moveCase", "POST", f"/v1/cases/{case_id}/state", body={"state": state})
