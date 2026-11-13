import { useEffect, useReducer, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ApiError, loadConversation, sendMessage, startConversation, type Session } from "./api";
import { chatReducer, emptyChat } from "./chatState";
import type { Chip, Intent } from "./sse";

const CONVERSATION_KEY = "hivemind.conversation";
const MAX_CHARS = 2000;

// buttons that already know what the customer wants, so no model has to read the message
const QUICK: { label: string; intent: Intent }[] = [
  { label: "Where is my order?", intent: { case_type: "where_is_my_order" } },
  { label: "My coupon didn't apply", intent: { case_type: "offers_coupons" } },
];

export function Chat({ session, onSignOut }: { session: Session; onSignOut: () => void }) {
  const [chat, dispatch] = useReducer(chatReducer, emptyChat);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const opened = useRef(false);
  const token = session.session_token;

  const open = async (fresh: boolean) => {
    setLoadError(null);
    try {
      const saved = fresh ? null : sessionStorage.getItem(CONVERSATION_KEY);
      const conversation = saved ? await loadConversation(token, saved) : await startConversation(token);
      sessionStorage.setItem(CONVERSATION_KEY, conversation.conversation_id);
      setConversationId(conversation.conversation_id);
      dispatch({ type: "loaded", messages: conversation.messages });
    } catch (problem) {
      if (problem instanceof ApiError && problem.status === 401) return onSignOut();
      if (problem instanceof ApiError && problem.status === 404 && !fresh) return open(true);
      setLoadError("Couldn't reach the HiveMind chat service. Is it running?");
    }
  };

  useEffect(() => {
    // once per sign-in, also when react runs effects twice in development
    if (opened.current) return;
    opened.current = true;
    void open(false);
  });

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [chat.messages]);

  const send = async (text: string, intent?: Intent) => {
    const clean = text.trim();
    if (!clean || chat.busy || !conversationId) return;
    const clientMsgId = crypto.randomUUID();
    dispatch({ type: "sent", text: clean, clientMsgId });
    try {
      await sendMessage(token, conversationId, { text: clean, clientMsgId, intent }, (event) =>
        dispatch({ type: "event", event }),
      );
    } catch (problem) {
      if (problem instanceof ApiError && problem.status === 401) return onSignOut();
      const busy = problem instanceof ApiError && problem.status === 409;
      dispatch({
        type: "failed",
        detail: busy ? "Still working on your last message. Try again in a moment." : "Couldn't send that. Try again.",
      });
    }
  };

  const sendDraft = () => {
    if (chat.busy || !conversationId || !draft.trim()) return;
    void send(draft);
    setDraft("");
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    sendDraft();
  };

  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendDraft();
    }
  };

  const last = chat.messages[chat.messages.length - 1];
  const chips: Chip[] = last && last.role === "assistant" && !chat.busy ? last.chips : [];

  return (
    <main className="chat">
      <header>
        <div>
          <h1>HiveMart support</h1>
          <p className="muted">
            Signed in as {session.name} ({session.customer_id})
          </p>
        </div>
        <div className="actions">
          <button type="button" onClick={() => void open(true)} disabled={chat.busy}>
            New chat
          </button>
          <button type="button" onClick={onSignOut}>
            Sign out
          </button>
        </div>
      </header>

      <p className="notice">
        You are chatting with HiveMind, an AI assistant, not a person. It reads your orders, deliveries and coupons to
        answer you. When it can't settle something itself, a person in customer care reviews it and the answer comes
        back here.
      </p>

      <section className="messages" aria-live="polite">
        {loadError && (
          <p className="error" role="alert">
            {loadError}{" "}
            <button type="button" onClick={() => void open(false)}>
              Retry
            </button>
          </p>
        )}
        {chat.messages.map((message) => (
          <div key={message.key} className={`bubble ${message.role}`}>
            {message.writing && !message.text ? (
              <>
                <span className="typing" aria-label="HiveMind is working on it">
                  <i />
                  <i />
                  <i />
                </span>
                {message.status && <span className="progress">{message.status}…</span>}
              </>
            ) : (
              message.text
            )}
          </div>
        ))}
        {chips.length > 0 && (
          <div className="chips">
            {chips.map((chip) => (
              <button key={chip.label} type="button" onClick={() => void send(chip.text, chip.intent)}>
                {chip.label}
              </button>
            ))}
          </div>
        )}
        {chat.error && (
          <p className="error" role="alert">
            {chat.error}
          </p>
        )}
        <div ref={bottom} />
      </section>

      <footer>
        <div className="quick">
          {QUICK.map((quick) => (
            <button
              key={quick.label}
              type="button"
              disabled={chat.busy || !conversationId}
              onClick={() => void send(quick.label, quick.intent)}
            >
              {quick.label}
            </button>
          ))}
        </div>
        <form onSubmit={submit}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value.slice(0, MAX_CHARS))}
            onKeyDown={onKey}
            placeholder="Type your message"
            rows={2}
            aria-label="Your message"
          />
          <button type="submit" disabled={chat.busy || !draft.trim() || !conversationId}>
            Send
          </button>
        </form>
      </footer>
    </main>
  );
}
