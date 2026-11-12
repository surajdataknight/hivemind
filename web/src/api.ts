// the two services the screen talks to: the business's fake sign-in (/shop) and the hivemind chat service (/api)

import { readEvents, type ChatEvent, type Intent, type ServerMessage } from "./sse";

export type SignInOption = { customer_id: string; name: string; hints: string[] };
export type Session = { session_token: string; customer_id: string; name: string; expires_at: string };
export type Conversation = { conversation_id: string; messages: ServerMessage[] };

export class ApiError extends Error {
  constructor(
    public status: number,
    detail: string,
  ) {
    super(detail);
  }
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // the body was not json; the status line is all there is
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const bearer = (token: string) => ({ Authorization: `Bearer ${token}` });

export async function signInOptions(): Promise<SignInOption[]> {
  const body = await json<{ customers: SignInOption[] }>(await fetch("/shop/v1/identity/sign-in-options"));
  return body.customers;
}

export async function signIn(customerId: string): Promise<Session> {
  const response = await fetch("/shop/v1/identity/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id: customerId }),
  });
  return json<Session>(response);
}

export async function startConversation(token: string): Promise<Conversation> {
  return json<Conversation>(await fetch("/api/v1/conversations", { method: "POST", headers: bearer(token) }));
}

export async function loadConversation(token: string, conversationId: string): Promise<Conversation> {
  return json<Conversation>(await fetch(`/api/v1/conversations/${conversationId}`, { headers: bearer(token) }));
}

export async function sendMessage(
  token: string,
  conversationId: string,
  message: { text: string; clientMsgId: string; intent?: Intent },
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const response = await fetch(`/api/v1/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { ...bearer(token), "Content-Type": "application/json" },
    body: JSON.stringify({ text: message.text, client_msg_id: message.clientMsgId, intent: message.intent ?? null }),
  });
  if (!response.ok || !response.body) {
    await json(response);
    throw new ApiError(response.status, "no reply stream");
  }
  let finished = false;
  await readEvents(response.body, (event) => {
    if (event.type === "done" || event.type === "error") finished = true;
    onEvent(event);
  });
  if (!finished) throw new ApiError(0, "the reply stopped before it was complete");
}
