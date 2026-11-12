// what the chat screen shows, as a pure function of what happened; no network in here

import type { ChatEvent, Chip, ServerMessage } from "./sse";

export type Message = {
  key: string;
  role: "customer" | "assistant";
  text: string;
  chips: Chip[];
  writing: boolean; // an assistant reply that is still arriving
};

export type ChatState = {
  messages: Message[];
  busy: boolean;
  error: string | null;
};

export type Action =
  | { type: "loaded"; messages: ServerMessage[] }
  | { type: "sent"; text: string; clientMsgId: string }
  | { type: "event"; event: ChatEvent }
  | { type: "failed"; detail: string };

export const emptyChat: ChatState = { messages: [], busy: false, error: null };

const fromServer = (m: ServerMessage): Message => ({
  key: `s${m.seq}`,
  role: m.role,
  text: m.text,
  chips: m.chips,
  writing: false,
});

function updateDraft(state: ChatState, change: (draft: Message) => Message): ChatState {
  const last = state.messages[state.messages.length - 1];
  if (!last || !last.writing) return state;
  return { ...state, messages: [...state.messages.slice(0, -1), change(last)] };
}

const withoutDraft = (messages: Message[]) => messages.filter((m) => !m.writing);

export function chatReducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "loaded":
      return { messages: action.messages.map(fromServer), busy: false, error: null };
    case "sent":
      return {
        busy: true,
        error: null,
        messages: [
          ...state.messages,
          { key: `c${action.clientMsgId}`, role: "customer", text: action.text, chips: [], writing: false },
          { key: `a${action.clientMsgId}`, role: "assistant", text: "", chips: [], writing: true },
        ],
      };
    case "event": {
      const event = action.event;
      if (event.type === "token") return updateDraft(state, (d) => ({ ...d, text: d.text + event.text }));
      if (event.type === "replace") return updateDraft(state, (d) => ({ ...d, text: event.text }));
      if (event.type === "done") {
        // the stored reply is the last word on what was said
        const kept = withoutDraft(state.messages);
        return { messages: [...kept, fromServer(event.message)], busy: false, error: null };
      }
      return { messages: withoutDraft(state.messages), busy: false, error: event.detail };
    }
    case "failed":
      return { messages: withoutDraft(state.messages), busy: false, error: action.detail };
  }
}
