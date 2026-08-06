import type { UsageStatus, UsageSummary } from "../../lib/api/types";
import { cn } from "../../lib/cn";

function find(usage: UsageStatus[], metric: string): UsageStatus | undefined {
  return usage.find((status) => status.metric === metric);
}

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(Math.round(value));
}

function Bar({ label, used, limit }: { label: string; used: number; limit: number | null }) {
  const ratio = limit ? Math.min(1, used / limit) : 0;
  const tone = ratio >= 1 ? "bg-destructive" : ratio >= 0.8 ? "bg-warning" : "bg-primary";
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="num-tabular text-muted-foreground">
          {formatCount(used)}
          {limit ? ` / ${formatCount(limit)}` : ""}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
        {limit ? (
          <div className={cn("h-full rounded-full transition-all", tone)} style={{ width: `${ratio * 100}%` }} />
        ) : (
          <div className="h-full w-full rounded-full bg-secondary" />
        )}
      </div>
    </div>
  );
}

/** The billing surface ADR-0004 exists to back: a plan that is real, and
 * numbers that are the same ones a run is actually admitted or refused
 * against — not a static badge next to an unconnected token count. */
export function UsagePanel({ usage }: { usage: UsageSummary | null }) {
  if (!usage) return null;

  const inputTokens = find(usage.usage, "llm_input_tokens");
  const outputTokens = find(usage.usage, "llm_output_tokens");
  const runs = find(usage.usage, "agent_run");

  const tokensUsed = (inputTokens?.used ?? 0) + (outputTokens?.used ?? 0);
  const tokensLimit =
    inputTokens?.limit != null && outputTokens?.limit != null
      ? inputTokens.limit + outputTokens.limit
      : null;

  return (
    <div className="border-t border-sidebar-border px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="label-eyebrow">Usage this month</span>
        <span className="rounded-full bg-primary-tint px-2 py-0.5 text-[11px] font-medium capitalize text-primary">
          {usage.plan.name}
        </span>
      </div>
      <div className="space-y-2">
        <Bar label="Tokens" used={tokensUsed} limit={tokensLimit} />
        {runs && <Bar label="Agent runs" used={runs.used} limit={runs.limit} />}
      </div>
    </div>
  );
}
