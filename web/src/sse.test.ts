import { describe, expect, it } from "vitest";
import { parseEvents, readEvents, type ChatEvent } from "./sse";

const token = (text: string) => `event: token\ndata: ${JSON.stringify({ type: "token", text })}\n\n`;

function streamOf(pieces: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      pieces.forEach((piece) => controller.enqueue(encoder.encode(piece)));
      controller.close();
    },
  });
}

describe("parseEvents", () => {
  it("returns the complete events and keeps the unfinished rest", () => {
    const text = token("Hello ") + token("there.") + "event: done\ndata: {\"type\":\"do";
    const { events, rest } = parseEvents(text);
    expect(events).toEqual([
      { type: "token", text: "Hello " },
      { type: "token", text: "there." },
    ]);
    expect(rest).toBe('event: done\ndata: {"type":"do');
  });

  it("reads windows line ends and ignores blocks without data", () => {
    const { events } = parseEvents("event: token\r\ndata: {\"type\":\"token\",\"text\":\"a\"}\r\n\r\n: keep-alive\n\n");
    expect(events).toEqual([{ type: "token", text: "a" }]);
  });

  it("keeps a rupee sign and a line break inside the text", () => {
    const { events } = parseEvents(token("₹499\nminimum"));
    expect(events).toEqual([{ type: "token", text: "₹499\nminimum" }]);
  });
});

describe("readEvents", () => {
  it("joins events that arrive cut in the middle, even inside a character", async () => {
    const whole = token("Your cart was ₹462. ") + token("The minimum is ₹499.");
    const bytes = new TextEncoder().encode(whole);
    const cut = bytes.indexOf(0xe2) + 1; // the middle of the first rupee sign
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, cut));
        controller.enqueue(bytes.slice(cut, cut + 40));
        controller.enqueue(bytes.slice(cut + 40));
        controller.close();
      },
    });
    const seen: ChatEvent[] = [];
    await readEvents(body, (event) => seen.push(event));
    expect(seen).toEqual([
      { type: "token", text: "Your cart was ₹462. " },
      { type: "token", text: "The minimum is ₹499." },
    ]);
  });

  it("delivers a last event that has no blank line after it", async () => {
    const seen: ChatEvent[] = [];
    await readEvents(streamOf([token("a"), token("b").trimEnd()]), (event) => seen.push(event));
    expect(seen.map((event) => (event.type === "token" ? event.text : ""))).toEqual(["a", "b"]);
  });
});
