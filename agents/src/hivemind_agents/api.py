"""the chat service's web api. the chat screen and the scenario panel talk only to this.

    POST /v1/conversations                      start a chat (session token in the Authorization header)
    GET  /v1/conversations/{id}                 the chat so far
    POST /v1/conversations/{id}/messages        send a message; the reply comes back as server-sent events
    GET  /v1/panel/cases                        the scenario panel: the latest cases (panel key in the header)
    GET  /v1/panel/cases/{case_id}              one case: plan, checks, findings, fact sheets, reads

the panel shows what hivemind found, the risk check included, so a customer's session token never opens it: it
needs the panel key (HM_PANEL_KEY in .env); with none set, the panel is off.
"""

import json
import logging
import queue
import secrets
import threading
from typing import Annotated, Iterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hivemind_agents.capabilities.identity import NotSignedIn
from hivemind_agents.capabilities.input_guard import MAX_MESSAGE_CHARS
from hivemind_agents.chat import Busy, ChatService, NotYours
from hivemind_agents.contracts import IntentHint
from hivemind_agents.panel import PanelReader

log = logging.getLogger("hivemind_agents")


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    client_msg_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
                               description="made by the screen for each send; a repeat gets the first reply again")
    intent: IntentHint | None = Field(default=None, description="set when the customer pressed a button")


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(service: ChatService, panel: PanelReader | None = None, panel_key: str | None = None) -> FastAPI:
    app = FastAPI(title="HiveMind chat service", version="0.2.0")

    def session_token(authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="send the session token as `Authorization: Bearer <token>`")
        return authorization.removeprefix("Bearer ")

    def refuse(exc: Exception) -> HTTPException:
        if isinstance(exc, NotSignedIn):
            return HTTPException(status_code=401, detail=str(exc))
        if isinstance(exc, NotYours):
            return HTTPException(status_code=404, detail="no such conversation")
        if isinstance(exc, Busy):
            return HTTPException(status_code=409, detail="still working on the previous message")
        return HTTPException(status_code=422, detail=str(exc))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/v1/conversations")
    def start_conversation(authorization: Annotated[str | None, Header()] = None) -> dict:
        try:
            return service.start_conversation(session_token(authorization))
        except NotSignedIn as exc:
            raise refuse(exc) from None

    @app.get("/v1/conversations/{conversation_id}")
    def get_conversation(conversation_id: str, authorization: Annotated[str | None, Header()] = None) -> dict:
        try:
            return service.conversation(session_token(authorization), conversation_id)
        except (NotSignedIn, NotYours) as exc:
            raise refuse(exc) from None

    @app.post("/v1/conversations/{conversation_id}/messages")
    def send_message(conversation_id: str, body: MessageIn,
                     authorization: Annotated[str | None, Header()] = None) -> StreamingResponse:
        token = session_token(authorization)
        try:
            service.check(token, conversation_id, body.text)
        except (NotSignedIn, NotYours, ValueError) as exc:
            raise refuse(exc) from None

        events: queue.Queue[dict | None] = queue.Queue()

        def work() -> None:
            # the whole run happens on this one thread, so its trace spans nest properly
            try:
                service.handle_message(token, conversation_id, body.text, body.client_msg_id, body.intent, events.put)
            except Busy:
                events.put({"type": "error", "code": "busy", "detail": "still working on the previous message"})
            except NotSignedIn as exc:
                events.put({"type": "error", "code": "not_signed_in", "detail": str(exc)})
            except Exception:  # the customer gets a plain line; the cause goes to the server log
                log.exception("chat run failed")
                events.put({"type": "error", "code": "failed", "detail": "something went wrong on our side"})
            finally:
                events.put(None)

        threading.Thread(target=work, name=f"chat-{conversation_id}", daemon=True).start()

        def stream() -> Iterator[str]:
            while (event := events.get()) is not None:
                yield sse(event)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def panel_reader(authorization: str | None) -> PanelReader:
        if panel is None or not panel_key:
            raise HTTPException(status_code=503, detail="the scenario panel is off: set HM_PANEL_KEY in .env")
        given = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
        if not secrets.compare_digest(given.encode(), panel_key.encode()):
            raise HTTPException(status_code=401, detail="send the panel key as `Authorization: Bearer <key>`")
        return panel

    @app.get("/v1/panel/cases")
    def panel_cases(limit: int = 30, authorization: Annotated[str | None, Header()] = None) -> dict:
        return {"cases": panel_reader(authorization).cases(limit)}

    @app.get("/v1/panel/cases/{case_id}")
    def panel_case(case_id: str, authorization: Annotated[str | None, Header()] = None) -> dict:
        found = panel_reader(authorization).case(case_id)
        if found is None:
            raise HTTPException(status_code=404, detail="no such case")
        return found

    return app
