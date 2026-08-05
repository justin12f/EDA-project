/**
 * Turns one agent event — live off the SSE stream, or replayed from
 * `GET /v1/threads/:id/events` — into the shape the chat panel renders.
 *
 * The same conversion either way is what makes a reopened thread look
 * identical to the one that was live a moment ago: both are just a list of
 * these events, read from a different source.
 */

import type { TranscriptItem } from "./api/types";

// Prefixed distinctly from ChatPanel's own counter (optimistic user bubbles,
// error bubbles): both land in the same list, and two independent counters
// producing the same "item-N" string would collide as React keys.
let counter = 0;
const nextId = () => `evt-${++counter}`;

export interface RawEvent {
  type: string;
  payload: Record<string, unknown>;
}

export function toTranscriptItem(event: RawEvent): TranscriptItem | null {
  switch (event.type) {
    case "message": {
      const text = String(event.payload.text ?? "");
      if (!text) return null;
      return event.payload.role === "user"
        ? { kind: "user", id: nextId(), text }
        : { kind: "assistant", id: nextId(), text };
    }
    case "tool_call": {
      const name = String(event.payload.name ?? "tool");
      const args = (event.payload.arguments as Record<string, unknown>) ?? {};
      return { kind: "tool_call", id: nextId(), name, args };
    }
    case "tool_result": {
      const name = String(event.payload.name ?? "tool");
      return { kind: "tool_result", id: nextId(), name, ok: Boolean(event.payload.ok) };
    }
    case "done": {
      const error = event.payload.error;
      return error ? { kind: "error", id: nextId(), text: String(error) } : null;
    }
    default:
      return null;
  }
}

/** A persisted thread's full history, oldest first, ready to render. */
export function replayTranscript(events: RawEvent[]): TranscriptItem[] {
  const items: TranscriptItem[] = [];
  for (const event of events) {
    const item = toTranscriptItem(event);
    if (item) items.push(item);
  }
  return items;
}
