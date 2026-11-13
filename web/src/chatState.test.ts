import { describe, expect, it } from "vitest";
import { chatReducer, emptyChat, type ChatState } from "./chatState";
import type { ServerMessage } from "./sse";

const stored = (seq: number, role: "customer" | "assistant", text: string, chips = []): ServerMessage => ({
  seq,
  role,
  text,
  kind: role === "customer" ? "message" : "order_status",
  case_id: null,
  chips,
  created_at: "2026-11-12T10:00:00+00:00",
});

const welcome = stored(1, "assistant", "Hi, I'm HiveMind, an AI assistant.");

function afterSend(): ChatState {
  const loaded = chatReducer(emptyChat, { type: "loaded", messages: [welcome] });
  return chatReducer(loaded, { type: "sent", text: "where is my order", clientMsgId: "m1" });
}

describe("chatReducer", () => {
  it("shows the stored chat when it loads", () => {
    const state = chatReducer(emptyChat, { type: "loaded", messages: [welcome] });
    expect(state.messages.map((m) => [m.role, m.text, m.writing])).toEqual([["assistant", welcome.text, false]]);
    expect(state.busy).toBe(false);
  });

  it("shows the customer's message at once, and a reply that is being written", () => {
    const state = afterSend();
    expect(state.busy).toBe(true);
    expect(state.messages.slice(1).map((m) => [m.role, m.text, m.writing])).toEqual([
      ["customer", "where is my order", false],
      ["assistant", "", true],
    ]);
  });

  it("adds each piece to the reply as it arrives", () => {
    let state = afterSend();
    state = chatReducer(state, { type: "event", event: { type: "token", text: "Your cart was ₹462. " } });
    state = chatReducer(state, { type: "event", event: { type: "token", text: "The minimum is ₹499." } });
    expect(state.messages[2].text).toBe("Your cart was ₹462. The minimum is ₹499.");
    expect(state.busy).toBe(true);
  });

  it("shows what is being looked at while the checks run, and none of it once the reply is stored", () => {
    let state = afterSend();
    const progress = (st: "started" | "finished", text: string) =>
      ({ type: "event", event: { type: "progress", state: st, text } }) as const;
    state = chatReducer(state, progress("started", "Checking your order details"));
    state = chatReducer(state, progress("started", "Reviewing delivery information"));
    state = chatReducer(state, progress("finished", "Checking your order details"));
    expect([state.messages[2].text, state.messages[2].status]).toEqual(["", "Reviewing delivery information"]);
    const final = stored(3, "assistant", "A person in customer care will review this.");
    state = chatReducer(state, { type: "event", event: { type: "done", message: final } });
    expect(state.messages.map((m) => m.status)).toEqual(["", "", ""]);
  });

  it("swaps the reply when the service replaces it", () => {
    let state = afterSend();
    state = chatReducer(state, { type: "event", event: { type: "token", text: "I'm sorry. " } });
    state = chatReducer(state, { type: "event", event: { type: "replace", text: "The minimum is ₹499." } });
    expect(state.messages[2].text).toBe("The minimum is ₹499.");
  });

  it("ends with the stored reply, whatever was streamed", () => {
    let state = afterSend();
    state = chatReducer(state, { type: "event", event: { type: "token", text: "Your order is on " } });
    const final = stored(3, "assistant", "Your order ORD-100360 is on its way.");
    state = chatReducer(state, { type: "event", event: { type: "done", message: final } });
    expect(state.busy).toBe(false);
    expect(state.messages.map((m) => m.text)).toEqual([welcome.text, "where is my order", final.text]);
    expect(state.messages.every((m) => !m.writing)).toBe(true);
  });

  it("drops the half-written reply and says so when the run fails", () => {
    let state = afterSend();
    state = chatReducer(state, { type: "event", event: { type: "token", text: "Your order" } });
    state = chatReducer(state, {
      type: "event",
      event: { type: "error", code: "failed", detail: "something went wrong on our side" },
    });
    expect(state.messages.map((m) => m.role)).toEqual(["assistant", "customer"]);
    expect(state.error).toBe("something went wrong on our side");
    expect(state.busy).toBe(false);
  });

  it("drops the half-written reply when the connection fails, and a new send clears the error", () => {
    let state = chatReducer(afterSend(), { type: "failed", detail: "Couldn't send that. Try again." });
    expect(state.error).toBe("Couldn't send that. Try again.");
    expect(state.messages).toHaveLength(2);
    state = chatReducer(state, { type: "sent", text: "again", clientMsgId: "m2" });
    expect(state.error).toBeNull();
  });

  it("ignores a stray piece when no reply is being written", () => {
    const loaded = chatReducer(emptyChat, { type: "loaded", messages: [welcome] });
    const state = chatReducer(loaded, { type: "event", event: { type: "token", text: "stray" } });
    expect(state.messages.map((m) => m.text)).toEqual([welcome.text]);
  });
});
