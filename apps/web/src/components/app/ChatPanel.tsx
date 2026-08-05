import { useEffect, useRef, useState } from "react";

import { streamRun, type AgentEvent } from "../../lib/api/client";
import type { Proposal, Source, TranscriptItem } from "../../lib/api/types";
import { MessageBubble } from "./MessageBubble";
import { ProposalCard } from "./ProposalCard";

let counter = 0;
const nextId = () => `item-${++counter}`;

export function ChatPanel({
  threadId,
  selectedSource,
  proposals,
  onProposalsChanged,
}: {
  threadId: string;
  selectedSource: Source | null;
  proposals: Proposal[];
  onProposalsChanged: () => void;
}) {
  const [items, setItems] = useState<TranscriptItem[]>([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items, sending]);

  async function send(prompt: string) {
    const text = prompt.trim();
    if (!text || sending) return;
    setDraft("");
    setSending(true);
    setItems((prev) => [...prev, { kind: "user", id: nextId(), text }]);

    try {
      await streamRun(
        { prompt: text, source_id: selectedSource?.id, thread_id: threadId },
        (event) => applyEvent(event, setItems),
      );
    } catch (err) {
      setItems((prev) => [
        ...prev,
        {
          kind: "error",
          id: nextId(),
          text: err instanceof Error ? err.message : "The agent did not respond",
        },
      ]);
    } finally {
      setSending(false);
      onProposalsChanged();
    }
  }

  return (
    <div className="flex h-full flex-1 flex-col">
      <header className="flex h-14 shrink-0 items-center border-b border-border px-6">
        <h2 className="text-[13px] font-semibold text-foreground">
          {selectedSource ? selectedSource.name : "General chat"}
        </h2>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-3">
          {items.length === 0 && (
            <EmptyState hasSource={!!selectedSource} onPrompt={send} />
          )}

          {items.map((item) => (
            <MessageBubble key={item.id} item={item} />
          ))}

          {sending && (
            <div className="flex items-center gap-2 pl-1 text-[12px] text-muted-foreground">
              <span className="ai-pulse h-1.5 w-1.5 rounded-full bg-ai" />
              thinking…
            </div>
          )}

          {proposals.map((proposal) => (
            <ProposalCard key={proposal.id} proposal={proposal} onDecided={onProposalsChanged} />
          ))}

          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-border px-6 py-4">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            send(draft);
          }}
          className="input-focus mx-auto flex max-w-2xl items-center gap-2 rounded-xl border border-input bg-background px-3 py-2"
        >
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder={
              selectedSource ? `Ask about ${selectedSource.name}…` : "Ask about your data…"
            }
            disabled={sending}
            className="h-8 flex-1 bg-transparent text-[13px] text-foreground placeholder:text-muted-foreground focus:outline-none"
          />
          <button
            type="submit"
            disabled={sending || !draft.trim()}
            className="h-8 shrink-0 rounded-lg bg-primary px-3.5 text-[13px] font-medium text-primary-foreground transition hover:brightness-110 disabled:opacity-40"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

function applyEvent(
  event: AgentEvent,
  setItems: React.Dispatch<React.SetStateAction<TranscriptItem[]>>,
): void {
  switch (event.type) {
    case "message": {
      // The user's own turn is echoed back over the stream (so a reopened
      // thread can replay it later) — already shown optimistically on send,
      // so only a non-"user" message is a new assistant bubble here.
      if (event.payload.role === "user") return;
      const text = String(event.payload.text ?? "");
      if (text) setItems((prev) => [...prev, { kind: "assistant", id: nextId(), text }]);
      return;
    }
    case "tool_call": {
      const name = String(event.payload.name ?? "tool");
      const args = (event.payload.arguments as Record<string, unknown>) ?? {};
      setItems((prev) => [...prev, { kind: "tool_call", id: nextId(), name, args }]);
      return;
    }
    case "tool_result": {
      const name = String(event.payload.name ?? "tool");
      setItems((prev) => [
        ...prev,
        { kind: "tool_result", id: nextId(), name, ok: Boolean(event.payload.ok) },
      ]);
      return;
    }
    case "done": {
      const error = event.payload.error;
      if (error) {
        setItems((prev) => [...prev, { kind: "error", id: nextId(), text: String(error) }]);
      }
      return;
    }
    default:
      return;
  }
}

function EmptyState({
  hasSource,
  onPrompt,
}: {
  hasSource: boolean;
  onPrompt: (text: string) => void;
}) {
  const suggestions = hasSource
    ? ["Profile this dataset", "Propose a cleaning pipeline", "Forecast the next 14 points"]
    : ["What data sources do I have?", "What have I learned about my data so far?"];

  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-4 pt-16 text-center">
      <div className="ai-pulse flex h-10 w-10 items-center justify-center rounded-xl bg-ai text-ai-foreground">
        <span className="text-sm font-semibold">L</span>
      </div>
      <p className="text-[13px] text-muted-foreground">
        {hasSource ? "Ask the agent to look at this dataset." : "Upload a source, then ask away."}
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => onPrompt(suggestion)}
            className="rounded-full border border-border bg-card px-3 py-1.5 text-[12px] text-foreground transition hover:border-primary hover:text-primary"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
