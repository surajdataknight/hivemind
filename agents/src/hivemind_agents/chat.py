"""the chat service: takes one customer message, runs the case graph, stores both sides, and reports events.

events, in order: `token` (a checked piece of the reply), `replace` (the reply so far is replaced by this text),
then exactly one `done` with the stored reply. the text in events is for the customer's eyes, with their own
personal data put back; what is stored and traced keeps the tokens.
"""

import copy
import threading
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from hivemind_agents import replies
from hivemind_agents.backend_client import BackendClient, BackendError, Http, NotFound
from hivemind_agents.capabilities.identity import IdentityVerifier, NotSignedIn
from hivemind_agents.capabilities.input_guard import MAX_MESSAGE_CHARS, PiiVault, guard
from hivemind_agents.capabilities.intake import HISTORY_TURNS
from hivemind_agents.case_store import CaseStore
from hivemind_agents.contracts import IntentHint, Reply
from hivemind_agents.graph import RECURSION_LIMIT, RunContext, build_graph
from hivemind_agents.llm import LLMUnavailable, ModelRouter
from hivemind_agents.store import ChatStore
from hivemind_agents.tracing import Tracer, set_output

Emit = Callable[[dict], None]


class NotYours(Exception):
    """no such conversation for this customer."""


class Busy(Exception):
    """the chat is still working on the previous message."""


@dataclass
class Deps:
    http: Http            # reaches the backend
    llm: ModelRouter
    store: ChatStore
    tracer: Tracer
    verifier: IdentityVerifier


class ChatService:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.graph = build_graph()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # sign-in and conversations

    def who(self, session_token: str) -> tuple[str, str]:
        return self.deps.verifier.verify_session(session_token)

    def start_conversation(self, session_token: str) -> dict:
        customer_id, session_id = self.who(session_token)
        conversation_id = self.deps.store.create_conversation(customer_id, session_id)
        return self.conversation(session_token, conversation_id)

    def conversation(self, session_token: str, conversation_id: str) -> dict:
        customer_id, _ = self.who(session_token)
        self._check_owner(conversation_id, customer_id)
        vault = self.deps.store.load_vault(conversation_id)
        rows = self.deps.store.messages(conversation_id)
        return {"conversation_id": conversation_id, "messages": [self._shown(row, vault) for row in rows]}

    def _check_owner(self, conversation_id: str, customer_id: str) -> None:
        if self.deps.store.owner(conversation_id) != customer_id:
            raise NotYours(conversation_id)

    @staticmethod
    def _shown(row: dict, vault: PiiVault) -> dict:
        """a stored message as the customer sees it."""
        return {"seq": row["seq"], "role": row["role"], "text": vault.restore(row["text"]), "kind": row["kind"],
                "case_id": row["case_id"], "chips": row["meta"].get("chips", []),
                "created_at": row["created_at"].isoformat()}

    # one message

    def check(self, session_token: str, conversation_id: str, text: str) -> None:
        """everything that can refuse a message before any work starts."""
        customer_id, _ = self.who(session_token)
        self._check_owner(conversation_id, customer_id)
        if not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            raise ValueError(f"a message needs 1 to {MAX_MESSAGE_CHARS} characters")

    def handle_message(self, session_token: str, conversation_id: str, text: str, client_msg_id: str,
                       hint: IntentHint | None, emit: Emit) -> dict:
        """runs one message to its reply. returns the `done` event it also emits."""
        self.check(session_token, conversation_id, text)
        lock = self._lock(conversation_id)
        if not lock.acquire(blocking=False):
            raise Busy(conversation_id)
        try:
            return self._handle(session_token, conversation_id, text, client_msg_id, hint, emit)
        finally:
            lock.release()

    def _lock(self, conversation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(conversation_id, threading.Lock())

    def _handle(self, session_token: str, conversation_id: str, text: str, client_msg_id: str,
                hint: IntentHint | None, emit: Emit) -> dict:
        deps, store = self.deps, self.deps.store
        customer_id, session_id = self.who(session_token)
        vault = store.load_vault(conversation_id)

        earlier = store.find_submit(conversation_id, client_msg_id)
        if earlier is not None:  # the same message sent twice: answer with what was already said
            _, reply_row = earlier
            if reply_row is None:
                raise Busy(conversation_id)
            done = {"type": "done", "message": self._shown(reply_row, vault), "repeat": True}
            emit(done)
            return done

        llm = copy.copy(deps.llm)
        llm.calls = 0
        with deps.tracer.run(conversation_id, customer_id, "") as run_span:
            state: dict[str, Any] = {}
            sent_seq: int | None = None

            def keep_customer_message(cleaned) -> int:
                """the customer's message is stored as soon as it is clean, before any model sees it."""
                store.save_vault(conversation_id, vault)
                run_span.set_attribute("input.value", cleaned.sanitised_text)
                return store.add_message(conversation_id, "customer", cleaned.sanitised_text, "message",
                                         client_msg_id=client_msg_id,
                                         meta={"flags": cleaned.flags, "pii_found": cleaned.pii_found})["seq"]

            def show(event: dict) -> None:
                emit({**event, "text": vault.restore(event["text"])})

            cases = CaseStore(store.conninfo, holder=f"{conversation_id}:{uuid4().hex[:12]}")
            try:
                with deps.tracer.capability("session_identity_verifier") as span:
                    identity, case_token = deps.verifier.mint_case_token(customer_id, session_id, conversation_id)
                    backend = BackendClient(deps.http, case_token, deps.tracer, ledger=cases.record_evidence)
                    try:
                        backend.me()  # the business must know this customer; otherwise nothing runs
                    except NotFound:
                        raise NotSignedIn("the business does not know this customer") from None
                    span.set_attribute("hivemind.scopes", identity.scopes)

                state = {"conversation_id": conversation_id, "identity": identity, "raw_text": text, "hint": hint,
                         "history": store.recent_turns(conversation_id, HISTORY_TURNS),
                         "pending": store.last_assistant_meta(conversation_id).get("pending")}
                context = RunContext(backend=backend, llm=llm, vault=vault, tracer=deps.tracer, cases=cases)
                for mode, chunk in self.graph.stream(state, {"recursion_limit": RECURSION_LIMIT}, context=context,
                                                     stream_mode=["custom", "values"]):
                    if mode == "custom":
                        show(chunk)
                        continue
                    state = chunk
                    if sent_seq is None and "guard" in state:
                        sent_seq = keep_customer_message(state["guard"])
                reply: Reply = state["reply"]
            except (BackendError, LLMUnavailable) as exc:
                run_span.set_attribute("hivemind.trouble", f"{type(exc).__name__}: {exc}")
                if sent_seq is None:
                    sent_seq = keep_customer_message(guard(text, vault))
                reply = Reply(kind="trouble", text=replies.TROUBLE)
                emit({"type": "replace", "text": reply.text})
            finally:
                cases.close()  # lets go of every case this run still holds

            reply.llm_calls = llm.calls
            meta = {**reply.meta, "reply_to": sent_seq, "llm_calls": llm.calls,
                    "chips": [c.model_dump() for c in reply.chips]}
            if "intake" in state:
                meta["intake"] = state["intake"].model_dump()
            stored = store.add_message(conversation_id, "assistant", reply.text, reply.kind, case_id=reply.case_id,
                                       meta=meta)
            run_span.set_attribute("hivemind.reply.kind", reply.kind)
            run_span.set_attribute("hivemind.llm_calls", llm.calls)
            if reply.case_id:
                run_span.set_attribute("hivemind.case_id", reply.case_id)
            set_output(run_span, reply.text)

        done = {"type": "done", "message": self._shown(stored, vault), "repeat": False,
                "case_state": reply.case_state, "llm_calls": llm.calls}
        emit(done)
        return done
