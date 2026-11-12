"""the chat service around the graph: its web api and event stream, its traces, and what it does when something
fails or a message arrives twice."""

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from hivemind_agents import replies
from hivemind_agents.api import create_app
from hivemind_agents.capabilities.identity import NotSignedIn
from hivemind_agents.chat import Busy, NotYours
from scripted import draft, issue

pytestmark = pytest.mark.db


def events_of(response) -> list[dict]:
    """the server-sent events of a response, in order."""
    assert response.headers["content-type"].startswith("text/event-stream")
    out = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        event = json.loads(data.removeprefix("data: "))
        assert name == f"event: {event['type']}"
        out.append(event)
    return out


@pytest.fixture
def api(service):
    return TestClient(create_app(service))


def signed_in(backend_http, customer_id: str) -> dict:
    token = backend_http.post("/v1/identity/sessions", json={"customer_id": customer_id}).json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


def send(api, headers, conversation_id, text, **extra):
    body = {"text": text, "client_msg_id": uuid4().hex, **extra}
    return api.post(f"/v1/conversations/{conversation_id}/messages", headers=headers, json=body)


# the web api


def test_a_chat_over_the_api(api, backend_http, data, model):
    order = data.live_order("OUT_FOR_DELIVERY")
    headers = signed_in(backend_http, order["customer_id"])
    opened = api.post("/v1/conversations", headers=headers).json()
    assert [m["kind"] for m in opened["messages"]] == ["welcome"]
    conversation_id = opened["conversation_id"]

    events = events_of(send(api, headers, conversation_id, "where is my order"))
    assert [e["type"] for e in events] == ["done"]  # a template reply arrives whole
    assert order["order_id"] in events[0]["message"]["text"] and events[0]["llm_calls"] == 0

    attempt = data.sole_attempt("cart below minimum", code="SAVE10")
    headers = signed_in(backend_http, attempt["customer_id"])
    conversation_id = api.post("/v1/conversations", headers=headers).json()["conversation_id"]
    model.intakes.append(draft(issues=[issue("offers_coupons", coupon_code="SAVE10")]))
    events = events_of(send(api, headers, conversation_id, "SAVE10 did not apply to my order"))
    assert [e["type"] for e in events][-1] == "done" and events[0]["type"] == "token" and len(events) >= 3
    said = "".join(e["text"] for e in events if e["type"] == "token").strip()
    assert said == events[-1]["message"]["text"] and "₹499" in said

    history = api.get(f"/v1/conversations/{conversation_id}", headers=headers).json()["messages"]
    assert [(m["role"], m["kind"]) for m in history] == [
        ("assistant", "welcome"), ("customer", "message"), ("assistant", "coupon_not_eligible")]


def test_a_button_press_over_the_api(api, backend_http, data):
    order = data.live_order("PACKING")
    headers = signed_in(backend_http, order["customer_id"])
    conversation_id = api.post("/v1/conversations", headers=headers).json()["conversation_id"]
    events = events_of(send(api, headers, conversation_id, "Where is my order?",
                            intent={"case_type": "where_is_my_order"}))
    assert events[-1]["message"]["kind"] == "order_status"


def test_the_api_refuses_before_any_work(api, backend_http, data):
    customer, other = data.orders[0]["customer_id"], data.orders[-1]["customer_id"]
    assert customer != other
    headers = signed_in(backend_http, customer)
    conversation_id = api.post("/v1/conversations", headers=headers).json()["conversation_id"]

    assert api.post("/v1/conversations").status_code == 401
    assert api.post("/v1/conversations", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert send(api, {}, conversation_id, "hi").status_code == 401
    # someone else's chat looks like no chat
    theirs = signed_in(backend_http, other)
    assert send(api, theirs, conversation_id, "hi").status_code == 404
    assert api.get(f"/v1/conversations/{conversation_id}", headers=theirs).status_code == 404
    assert send(api, headers, "conv_does_not_exist", "hi").status_code == 404
    # malformed messages
    assert send(api, headers, conversation_id, "").status_code == 422
    assert send(api, headers, conversation_id, "x" * 2001).status_code == 422
    assert send(api, headers, conversation_id, "   ").status_code == 422
    bad_id = api.post(f"/v1/conversations/{conversation_id}/messages", headers=headers,
                      json={"text": "hi", "client_msg_id": "short"})
    assert bad_id.status_code == 422
    assert send(api, headers, conversation_id, "hi", intent={"case_type": "free_money"}).status_code == 422
    assert len(api.get(f"/v1/conversations/{conversation_id}", headers=headers).json()["messages"]) == 1


# a message sent twice, and two at once


def test_the_same_message_twice_gets_the_first_reply_and_runs_once(chat_as, data, cases):
    order = data.live_order("OUT_FOR_DELIVERY")
    chat = chat_as(order["customer_id"])
    first = chat.say("where is my order", client_msg_id="same-id-0001")
    again = chat.say("where is my order", client_msg_id="same-id-0001")
    assert again["repeat"] is True and again["message"] == first["message"]
    assert len(chat.stored()) == 3 and len(cases()) == 1


def test_one_message_at_a_time_per_chat(chat_as, data, service):
    chat = chat_as(data.orders[0]["customer_id"])
    lock = service._lock(chat.conversation_id)
    lock.acquire()
    try:
        with pytest.raises(Busy):
            chat.say("hi")
    finally:
        lock.release()
    assert chat.say("hi")["message"]["kind"] == "greeting"


def test_a_chat_belongs_to_its_customer(chat_as, service, backend_http, data):
    chat = chat_as(data.orders[0]["customer_id"])
    other = backend_http.post("/v1/identity/sessions",
                              json={"customer_id": data.orders[-1]["customer_id"]}).json()["session_token"]
    with pytest.raises(NotYours):
        service.conversation(other, chat.conversation_id)
    with pytest.raises(NotYours):
        service.handle_message(other, chat.conversation_id, "hi", uuid4().hex, None, lambda e: None)
    with pytest.raises(NotSignedIn):
        service.handle_message("not-a-token", chat.conversation_id, "hi", uuid4().hex, None, lambda e: None)


# when something fails


def test_no_model_means_no_guess(chat_as, data, model, cases):
    """the model is down: a message only a model can read gets the trouble reply; rules still work."""
    model.down = True
    order = data.live_order("OUT_FOR_DELIVERY")
    chat = chat_as(order["customer_id"])
    done = chat.say("my coupon did not apply and I am annoyed")
    assert done["message"]["kind"] == "trouble" and done["message"]["text"] == replies.TROUBLE
    assert cases() == []
    assert [m["role"] for m in chat.stored()] == ["assistant", "customer", "assistant"]  # the message is kept
    assert chat.say("where is my order")["message"]["kind"] == "order_status"


def test_no_model_for_the_reply_means_the_template(chat_as, data, model, cases):
    attempt = data.sole_attempt("cart below minimum", code="SAVE10")
    chat = chat_as(attempt["customer_id"])

    model.intakes.append(draft(issues=[issue("offers_coupons", coupon_code="SAVE10")]))
    model.replies.append(RuntimeError("429 quota used up"))  # the model answers the intake, then runs out
    done = chat.say("SAVE10 did not apply")
    assert done["message"]["kind"] == "coupon_not_eligible" and "₹499" in done["message"]["text"]
    assert chat.stored()[-1]["meta"]["communicator"]["source"] == "template"
    assert cases()[0]["state"] == "RESOLVED"


def test_a_model_that_misstates_an_amount_is_overruled(chat_as, data, model):
    attempt = data.sole_attempt("cart below minimum", code="SAVE10")
    chat = chat_as(attempt["customer_id"])
    model.intakes.append(draft(issues=[issue("offers_coupons", coupon_code="SAVE10")]))
    model.replies += ["Good news! I have refunded ₹5,000 to you.", "You will get ₹250 back tomorrow."]
    done = chat.say("SAVE10 did not apply")
    text = done["message"]["text"]
    assert "5,000" not in text and "250" not in text and "tomorrow" not in text and "₹499" in text
    assert done["llm_calls"] == 3  # intake, and two tries at the reply
    assert chat.streamed_text() == text
    assert all("5,000" not in e.get("text", "") for e in chat.events)


def test_an_intake_answer_that_is_not_the_form_gets_one_repair_try(chat_as, data, model):
    chat = chat_as(data.orders[0]["customer_id"])
    model.intakes += ["I think the customer is upset.", draft(scope="out_of_scope")]
    done = chat.say("please change my address")
    assert done["message"]["kind"] == "out_of_scope" and done["llm_calls"] == 2
    assert "was not valid" in model.calls[1]["prompt"]

    model.intakes += ["nope", '{"scope": "refund everything"}']
    assert chat.say("please change my address again")["message"]["kind"] == "trouble"


def test_the_backend_being_down_is_trouble_not_a_guess(service, chat_as, data):
    order = data.live_order("OUT_FOR_DELIVERY")
    chat = chat_as(order["customer_id"])

    class Down:
        def request(self, *args, **kwargs):
            raise ConnectionError("refused")

    working = service.deps.http
    service.deps.http = Down()
    try:
        done = chat.say("where is my order")
    finally:
        service.deps.http = working
    assert done["message"]["kind"] == "trouble"
    assert chat.stored()[1]["text"] == "where is my order"


# what the models are shown


def test_the_model_sees_tokens_and_no_internal_data(chat_as, data, model):
    attempt = data.sole_attempt("cart below minimum", code="SAVE10")
    chat = chat_as(attempt["customer_id"])
    model.intakes.append(draft(issues=[issue("offers_coupons", coupon_code="SAVE10")]))
    chat.say("SAVE10 did not apply, call me on 9876543210")
    reading, writing = model.calls
    assert reading["kind"] == "generate" and writing["kind"] == "stream"
    for call in model.calls:
        assert call["model"] == "gemini-3.5-flash-lite"
        assert "9876543210" not in call["prompt"] and "<PHONE_1>" in call["prompt"]
        assert attempt["customer_id"] not in call["prompt"] and "ev_" not in call["prompt"]
        assert "CASE-" not in call["prompt"]
    assert attempt["order_id"] in reading["prompt"]  # the intake sees the customer's recent order ids
    assert "what the check found:" in writing["prompt"] and "the offer's terms" in writing["prompt"]


# traces


def spans_by_name(spans):
    out = {}
    for span in spans.get_finished_spans():
        out.setdefault(span.name, []).append(span)
    return out


def test_where_is_my_order_is_one_trace_with_no_model_span(chat_as, data, spans):
    order = data.live_order("OUT_FOR_DELIVERY")
    chat = chat_as(order["customer_id"])
    chat.say("where is my order")
    finished = spans.get_finished_spans()
    assert len({s.context.trace_id for s in finished}) == 1
    named = spans_by_name(spans)
    (root,) = named["case_run"]
    assert root.parent is None and root.attributes["session.id"] == chat.conversation_id
    assert root.attributes["input.value"] == "where is my order" and order["order_id"] in root.attributes["output.value"]
    assert root.attributes["hivemind.llm_calls"] == 0 and root.attributes["hivemind.reply.kind"] == "order_status"
    for capability in ("session_identity_verifier", "input_guard", "intake_triage", "triage", "case_manager",
                       "order_status"):
        assert capability in named, capability
    assert not [s for s in finished if s.attributes["openinference.span.kind"] == "LLM"]
    tools = {s.attributes["tool.name"] for s in finished if s.attributes["openinference.span.kind"] == "TOOL"}
    assert tools == {"getMe", "listMyOrders", "openCase", "moveCase", "getOrder", "getOrderTrip", "getClock"}
    order_read = named["execute_tool getOrder"][0]
    assert order_read.parent.span_id == named["order_status"][0].context.span_id
    assert order_read.attributes["hivemind.evidence_id"].startswith("ev_")
    assert all(s.status.is_ok for s in finished)


def test_a_coupon_chat_traces_two_model_calls_with_their_tokens(chat_as, data, model, spans):
    attempt = data.sole_attempt("cart below minimum", code="SAVE10")
    model.intakes.append(draft(issues=[issue("offers_coupons", coupon_code="SAVE10")]))
    chat_as(attempt["customer_id"]).say("SAVE10 did not apply")
    finished = spans.get_finished_spans()
    assert len({s.context.trace_id for s in finished}) == 1
    named = spans_by_name(spans)
    llm = [s for s in finished if s.attributes["openinference.span.kind"] == "LLM"]
    assert [s.name for s in llm] == ["chat gemini-3.5-flash-lite"] * 2
    assert [s.attributes["hivemind.job"] for s in llm] == ["intake", "communicator"]
    for span in llm:
        assert span.attributes["gen_ai.request.model"] == span.attributes["llm.model_name"] == "gemini-3.5-flash-lite"
        assert span.attributes["gen_ai.usage.input_tokens"] == span.attributes["llm.token_count.prompt"] > 0
        assert span.attributes["gen_ai.usage.output_tokens"] > 0 and span.attributes["output.value"]
    assert llm[0].parent.span_id == named["intake_triage"][0].context.span_id
    assert llm[1].parent.span_id == named["customer_communicator"][0].context.span_id
    for capability in ("promotions_investigator", "policy_knowledge_provider", "customer_communicator"):
        assert named[capability][0].parent.span_id is not None
    assert named["case_run"][0].attributes["hivemind.llm_calls"] == 2


def test_a_backend_call_carries_the_case_token_the_trace_and_the_case(service, chat_as, data):
    seen = []
    working = service.deps.http

    class Recorder:
        def request(self, method, url, **kwargs):
            seen.append((method, url, kwargs["headers"]))
            return working.request(method, url, **kwargs)

    order = data.live_order("OUT_FOR_DELIVERY")
    chat = chat_as(order["customer_id"])
    service.deps.http = Recorder()
    try:
        done = chat.say("where is my order")
    finally:
        service.deps.http = working
    assert all(h["Authorization"].startswith("Bearer ") and h["traceparent"].startswith("00-") for _, _, h in seen)
    assert len({h["traceparent"].split("-")[1] for _, _, h in seen}) == 1  # one trace for the whole run
    after_open = [h for _, url, h in seen if "/trip" in url or url.endswith("/state")]
    assert after_open and all(h["X-Case-Id"] == done["message"]["case_id"] for h in after_open)
    assert not any(order["customer_id"] in url for _, url, _ in seen)  # the customer is never an argument
