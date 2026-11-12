"""the chat memory in postgres (agent schema): conversations, the transcript, and the pii vault.

the transcript holds the token form of every message (<PHONE_1>, never the number). the vault maps tokens back
and is read only here, to show a customer their own words.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from hivemind_agents import replies
from hivemind_agents.capabilities.input_guard import PiiVault
from hivemind_agents.contracts import Turn


class ChatStore:
    def __init__(self, conninfo: str):
        self.conninfo = conninfo

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.conninfo, row_factory=dict_row)

    def create_conversation(self, customer_id: str, session_id: str) -> str:
        """a new chat, opened with the welcome that says hivemind is an ai."""
        conversation_id = "conv_" + uuid4().hex[:20]
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            conn.execute(
                "insert into agent.conversations (conversation_id, customer_id, session_id, started_at) "
                "values (%s, %s, %s, %s)", (conversation_id, customer_id, session_id, now))
            conn.execute(
                "insert into agent.messages (conversation_id, seq, role, text, kind, created_at) "
                "values (%s, 1, 'assistant', %s, 'welcome', %s)", (conversation_id, replies.WELCOME, now))
        return conversation_id

    def owner(self, conversation_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("select customer_id from agent.conversations where conversation_id = %s",
                               (conversation_id,)).fetchone()
        return row["customer_id"] if row else None

    def messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            return conn.execute(
                "select seq, role, text, kind, case_id, client_msg_id, meta, created_at from agent.messages "
                "where conversation_id = %s order by seq", (conversation_id,)).fetchall()

    def recent_turns(self, conversation_id: str, limit: int) -> list[Turn]:
        with self._connect() as conn:
            rows = conn.execute(
                "select role, text from agent.messages where conversation_id = %s order by seq desc limit %s",
                (conversation_id, limit)).fetchall()
        return [Turn(role=r["role"], text=r["text"]) for r in reversed(rows)]

    def last_assistant_meta(self, conversation_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "select role, meta from agent.messages where conversation_id = %s order by seq desc limit 1",
                (conversation_id,)).fetchone()
        return row["meta"] if row and row["role"] == "assistant" else {}

    def find_submit(self, conversation_id: str, client_msg_id: str) -> tuple[int, dict | None] | None:
        """an earlier submit of the same message: (its seq, the reply to it if there is one yet)."""
        with self._connect() as conn:
            sent = conn.execute(
                "select seq from agent.messages where conversation_id = %s and client_msg_id = %s",
                (conversation_id, client_msg_id)).fetchone()
            if sent is None:
                return None
            reply = conn.execute(
                "select seq, role, text, kind, case_id, meta, created_at from agent.messages "
                "where conversation_id = %s and role = 'assistant' and (meta ->> 'reply_to')::int = %s",
                (conversation_id, sent["seq"])).fetchone()
        return sent["seq"], reply

    def add_message(self, conversation_id: str, role: str, text: str, kind: str, *, case_id: str | None = None,
                    client_msg_id: str | None = None, meta: dict[str, Any] | None = None) -> dict:
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            # the conversation row is the lock: two writers cannot take the same seq
            conn.execute("select 1 from agent.conversations where conversation_id = %s for update",
                         (conversation_id,))
            return conn.execute(
                "insert into agent.messages (conversation_id, seq, role, text, kind, case_id, client_msg_id, meta, "
                "created_at) select %s, coalesce(max(seq), 0) + 1, %s, %s, %s, %s, %s, %s, %s from agent.messages "
                "where conversation_id = %s returning seq, role, text, kind, case_id, client_msg_id, meta, created_at",
                (conversation_id, role, text, kind, case_id, client_msg_id, Jsonb(meta or {}), now,
                 conversation_id)).fetchone()

    def load_vault(self, conversation_id: str) -> PiiVault:
        with self._connect() as conn:
            rows = conn.execute("select token, kind, value from agent.pii_vault where conversation_id = %s",
                                (conversation_id,)).fetchall()
        return PiiVault({r["token"]: (r["kind"], r["value"]) for r in rows})

    def save_vault(self, conversation_id: str, vault: PiiVault) -> None:
        if not vault.new:
            return
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            for token, (kind, value) in vault.new.items():
                conn.execute(
                    "insert into agent.pii_vault (conversation_id, token, kind, value, created_at) "
                    "values (%s, %s, %s, %s, %s) on conflict do nothing",
                    (conversation_id, token, kind, value, now))
        vault.new.clear()
