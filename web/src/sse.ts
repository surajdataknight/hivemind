// server-sent events, read from a fetch body: EventSource cannot post, so the stream is parsed here

export type ChatEvent =
  | { type: "progress"; state: "started" | "finished"; text: string }
  | { type: "token"; text: string }
  | { type: "replace"; text: string }
  | { type: "done"; message: ServerMessage }
  | { type: "error"; code: string; detail: string };

export type Intent = { case_type: string; order_id?: string | null; coupon_code?: string | null; items?: string[] };
export type Chip = { label: string; text: string; intent: Intent };

export type ServerMessage = {
  seq: number;
  role: "customer" | "assistant";
  text: string;
  kind: string;
  case_id: string | null;
  chips: Chip[];
  created_at: string;
};

// takes the text received so far; returns the complete events in it and the unfinished rest
export function parseEvents(buffer: string): { events: ChatEvent[]; rest: string } {
  const blocks = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    const data = block
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (data) events.push(JSON.parse(data) as ChatEvent);
  }
  return { events, rest };
}

export async function readEvents(body: ReadableStream<Uint8Array>, onEvent: (event: ChatEvent) => void) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseEvents(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(onEvent);
  }
  parseEvents(buffer + "\n\n").events.forEach(onEvent);
}
