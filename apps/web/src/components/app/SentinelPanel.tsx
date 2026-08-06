import { useEffect, useState } from "react";

import { apiGet, apiPut } from "../../lib/api/client";
import type { DriftEvent, Schedule } from "../../lib/api/types";

const STATUS_LABEL: Record<DriftEvent["status"], string> = {
  detected: "Detected",
  diagnosing: "Diagnosing…",
  proposed: "Proposed",
  resolved: "Resolved",
  dismissed: "Needs a look",
};

const STATUS_DOT: Record<DriftEvent["status"], string> = {
  detected: "bg-warning",
  diagnosing: "bg-ai",
  proposed: "bg-ai",
  resolved: "bg-success",
  dismissed: "bg-destructive",
};

function driftSummary(event: DriftEvent): string {
  const schema = event.details.schema_changes ?? [];
  if (schema.length) return schema.map((change) => change.detail).join("; ");
  const nulls = event.details.null_rate_shifts ?? [];
  if (nulls.length) {
    return nulls
      .map(
        (shift) =>
          `'${shift.column}' nulls ${(shift.previous * 100).toFixed(1)}% → ${(shift.current * 100).toFixed(1)}%`,
      )
      .join("; ");
  }
  return "No detail recorded.";
}

/**
 * ADR-0008's one new surface: a toggle for the source's scheduled drift scan,
 * and the feed of what it has found. Detection and diagnosis both run off the
 * request path (`services/worker`) — this panel only reads their results and
 * writes the opt-in flags a person controls.
 */
export function SentinelPanel({ sourceId }: { sourceId: string }) {
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [events, setEvents] = useState<DriftEvent[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    setSchedule(null);
    setEvents([]);
    setExpanded(false);

    apiGet<Schedule>(`/v1/sources/${sourceId}/schedule`)
      .then((result) => active && setSchedule(result))
      .catch(() => {});
    apiGet<{ events: DriftEvent[] }>(`/v1/sources/${sourceId}/drift-events`)
      .then((result) => active && setEvents(result.events))
      .catch(() => {});

    return () => {
      active = false;
    };
  }, [sourceId]);

  async function update(next: { enabled?: boolean; auto_apply_enabled?: boolean }) {
    if (!schedule || busy) return;
    setBusy(true);
    try {
      const updated = await apiPut<Schedule>(`/v1/sources/${sourceId}/schedule`, {
        enabled: next.enabled ?? schedule.enabled,
        auto_apply_enabled: next.auto_apply_enabled ?? schedule.auto_apply_enabled,
      });
      setSchedule(updated);
    } catch {
      // The toggle just snaps back to its last known state below — a failed
      // write here is not worth a dedicated error banner.
    } finally {
      setBusy(false);
    }
  }

  if (!schedule) return null;

  return (
    <div className="border-b border-border px-6 py-2.5">
      <div className="mx-auto flex max-w-2xl items-center gap-3">
        <button
          onClick={() => update({ enabled: !schedule.enabled })}
          disabled={busy}
          className={`flex h-6 shrink-0 items-center gap-1.5 rounded-full px-2.5 text-[11px] font-medium transition disabled:opacity-60 ${
            schedule.enabled ? "bg-ai-tint text-ai" : "bg-muted text-muted-foreground"
          }`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${schedule.enabled ? "bg-ai" : "bg-muted-foreground"}`} />
          Sentinel {schedule.enabled ? "on" : "off"}
        </button>

        {schedule.enabled && (
          <label className="flex shrink-0 items-center gap-1.5 text-[11px] text-muted-foreground">
            <input
              type="checkbox"
              checked={schedule.auto_apply_enabled}
              disabled={busy}
              onChange={(event) => update({ auto_apply_enabled: event.target.checked })}
              className="h-3 w-3 accent-ai"
            />
            Auto-apply safe fixes
          </label>
        )}

        {events.length > 0 && (
          <button
            onClick={() => setExpanded((value) => !value)}
            className="ml-auto shrink-0 text-[11px] font-medium text-ai hover:underline"
          >
            {expanded ? "Hide" : `${events.length} drift event${events.length === 1 ? "" : "s"}`}
          </button>
        )}
      </div>

      {expanded && (
        <ul className="mx-auto mt-2 flex max-w-2xl flex-col gap-1.5">
          {events.map((event) => (
            <li
              key={event.id}
              className="flex items-start gap-2 rounded-lg bg-secondary px-2.5 py-1.5 text-[12px]"
            >
              <span
                className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${STATUS_DOT[event.status] ?? "bg-muted-foreground"}`}
              />
              <span className="min-w-0 flex-1">
                <span className="block text-foreground">{driftSummary(event)}</span>
                <span className="block text-[11px] text-muted-foreground">
                  {STATUS_LABEL[event.status] ?? event.status} ·{" "}
                  {new Date(event.occurred_at).toLocaleString()}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
