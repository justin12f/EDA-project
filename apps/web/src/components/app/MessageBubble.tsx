import type { TranscriptItem } from "../../lib/api/types";

export function MessageBubble({ item }: { item: TranscriptItem }) {
  switch (item.kind) {
    case "user":
      return (
        <div className="flex justify-end">
          <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-[13px] text-primary-foreground">
            {item.text}
          </div>
        </div>
      );

    case "assistant":
      return (
        <div className="flex justify-start">
          <div className="max-w-[75%] rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-2.5 text-[13px] leading-relaxed text-foreground whitespace-pre-wrap">
            {item.text}
          </div>
        </div>
      );

    case "tool_call":
      return (
        <div className="flex items-center gap-2 pl-1 text-[12px] text-muted-foreground">
          <span className="ai-pulse h-1.5 w-1.5 rounded-full bg-ai" />
          <span className="font-mono">{item.name}</span>
          <span className="text-muted-foreground/70">{summarizeArgs(item.args)}</span>
        </div>
      );

    case "tool_result":
      return (
        <div className="flex items-center gap-2 pl-1 text-[12px] text-muted-foreground">
          <span className={item.ok ? "text-success" : "text-destructive"}>
            {item.ok ? "✓" : "✗"}
          </span>
          <span className="font-mono">{item.name}</span>
        </div>
      );

    case "error":
      return (
        <div className="rounded-lg border border-destructive-soft bg-destructive-tint px-3 py-2 text-[13px] text-destructive">
          {item.text}
        </div>
      );
  }
}

function summarizeArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args).filter(([key]) => key !== "steps");
  if (entries.length === 0) return "";
  return entries.map(([key, value]) => `${key}=${String(value).slice(0, 24)}`).join(" ");
}
