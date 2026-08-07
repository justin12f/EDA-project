import { useEffect, useState } from "react";

import { apiDelete, apiGet, apiPut } from "../../lib/api/client";
import type { BaselinesResponse, ColumnBaseline, DriftEvent, Schedule } from "../../lib/api/types";

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

const BASELINE_KIND_LABEL: Record<ColumnBaseline["baseline_kind"], string> = {
  numeric: "numeric",
  categorical: "categorical",
  datetime: "datetime",
  freeform_string: "free text",
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

  // ADR-0013: this column's own calibrated band, not the flat threshold above.
  const findings = event.details.findings ?? [];
  if (findings.length) {
    return findings
      .map((finding) => {
        if (finding.shift_kind === "new_category") {
          return `'${finding.column}' — new category '${finding.category}' at ${((finding.current_share ?? 0) * 100).toFixed(0)}%`;
        }
        if (finding.shift_kind === "share_shift") {
          return `'${finding.column}' — '${finding.category}' ${((finding.baseline_share ?? 0) * 100).toFixed(0)}% → ${((finding.current_share ?? 0) * 100).toFixed(0)}%`;
        }
        return `'${finding.column}' outside its usual range (${(finding.observed ?? 0).toFixed(2)}, usually ~${(finding.baseline_mean ?? 0).toFixed(2)})`;
      })
      .join("; ");
  }

  if (event.kind === "volume_freshness" && event.details.observed_delta !== undefined) {
    const delta = event.details.observed_delta;
    const usual = (event.details.baseline_mean ?? 0).toFixed(0);
    return delta <= 0
      ? `Row count stopped growing (${event.details.row_count} rows, usually +${usual}/scan)`
      : `Row count moved unusually (${delta > 0 ? "+" : ""}${delta.toFixed(0)}, usually ~${usual}/scan)`;
  }

  return "No detail recorded.";
}

/**
 * ADR-0008's one new surface: a toggle for the source's scheduled drift scan,
 * and the feed of what it has found. Detection and diagnosis both run off the
 * request path (`services/worker`) — this panel only reads their results and
 * writes the opt-in flags a person controls.
 *
 * ADR-0013 adds a second, collapsed-by-default disclosure: what each
 * column's own calibrated band currently looks like, and the manual
 * override escape hatch — deliberately not the primary view, matching the
 * ADR's own "never required" framing for the override.
 */
export function SentinelPanel({ sourceId }: { sourceId: string }) {
  const [schedule, setSchedule] = useState<Schedule | null>(null);
  const [events, setEvents] = useState<DriftEvent[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState(false);

  const [baselines, setBaselines] = useState<BaselinesResponse | null>(null);
  const [baselinesExpanded, setBaselinesExpanded] = useState(false);
  const [overrideDrafts, setOverrideDrafts] = useState<Record<string, string>>({});

  async function refetchBaselines() {
    const result = await apiGet<BaselinesResponse>(`/v1/sources/${sourceId}/baselines`);
    setBaselines(result);
  }

  useEffect(() => {
    let active = true;
    setSchedule(null);
    setEvents([]);
    setExpanded(false);
    setBaselines(null);
    setBaselinesExpanded(false);
    setOverrideDrafts({});

    apiGet<Schedule>(`/v1/sources/${sourceId}/schedule`)
      .then((result) => active && setSchedule(result))
      .catch(() => {});
    apiGet<{ events: DriftEvent[] }>(`/v1/sources/${sourceId}/drift-events`)
      .then((result) => active && setEvents(result.events))
      .catch(() => {});
    apiGet<BaselinesResponse>(`/v1/sources/${sourceId}/baselines`)
      .then((result) => active && setBaselines(result))
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

  async function setOverride(column: string, kind: ColumnBaseline["baseline_kind"]) {
    const raw = overrideDrafts[column];
    const value = raw === undefined || raw === "" ? undefined : Number(raw);
    if (value === undefined || Number.isNaN(value)) return;
    const body = kind === "categorical" ? { deviation_threshold: value } : { z: value };
    try {
      await apiPut(`/v1/sources/${sourceId}/baselines/${column}/override`, body);
      await refetchBaselines();
    } catch {
      // The field just keeps its typed value — same reasoning as the
      // schedule toggle above not getting a dedicated error banner.
    }
  }

  async function clearOverride(column: string) {
    try {
      await apiDelete(`/v1/sources/${sourceId}/baselines/${column}/override`);
      setOverrideDrafts((drafts) => {
        const next = { ...drafts };
        delete next[column];
        return next;
      });
      await refetchBaselines();
    } catch {
      // Same as above.
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

        <div className="ml-auto flex shrink-0 items-center gap-3">
          {baselines && baselines.columns.length > 0 && (
            <button
              onClick={() => setBaselinesExpanded((value) => !value)}
              className="text-[11px] font-medium text-muted-foreground hover:underline"
            >
              {baselinesExpanded
                ? "Hide baselines"
                : `${baselines.columns.length} column baseline${baselines.columns.length === 1 ? "" : "s"}`}
            </button>
          )}
          {events.length > 0 && (
            <button
              onClick={() => setExpanded((value) => !value)}
              className="text-[11px] font-medium text-ai hover:underline"
            >
              {expanded ? "Hide" : `${events.length} drift event${events.length === 1 ? "" : "s"}`}
            </button>
          )}
        </div>
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

      {baselinesExpanded && baselines && (
        <div className="mx-auto mt-2 max-w-2xl">
          <p className="mb-1.5 text-[11px] text-muted-foreground">
            Calibrated automatically from this source's own history — no setup required. A column
            needs {baselines.min_sample_size} scans before its band is trusted; an override (never
            required) widens or narrows it for one column.
          </p>
          <ul className="flex flex-col gap-1.5">
            {baselines.columns.map((column) => (
              <li
                key={column.column}
                className="flex items-center gap-2 rounded-lg bg-secondary px-2.5 py-1.5 text-[12px]"
              >
                <span
                  className={`h-1.5 w-1.5 shrink-0 rounded-full ${column.calibrated ? "bg-success" : "bg-muted-foreground"}`}
                  title={column.calibrated ? "Calibrated" : "Still calibrating"}
                />
                <span className="min-w-0 flex-1 truncate text-foreground">{column.column}</span>
                <span className="shrink-0 text-[11px] text-muted-foreground">
                  {BASELINE_KIND_LABEL[column.baseline_kind]} · {column.sample_size} scan
                  {column.sample_size === 1 ? "" : "s"}
                </span>
                <input
                  type="number"
                  placeholder={column.baseline_kind === "categorical" ? "sensitivity" : "z"}
                  value={overrideDrafts[column.column] ?? ""}
                  onChange={(event) =>
                    setOverrideDrafts((drafts) => ({ ...drafts, [column.column]: event.target.value }))
                  }
                  className="h-6 w-16 shrink-0 rounded border border-input bg-background px-1.5 text-[11px] text-foreground"
                />
                <button
                  onClick={() => setOverride(column.column, column.baseline_kind)}
                  className="shrink-0 text-[11px] font-medium text-ai hover:underline"
                >
                  Set
                </button>
                {column.override && (
                  <button
                    onClick={() => clearOverride(column.column)}
                    className="shrink-0 text-[11px] font-medium text-muted-foreground hover:underline"
                  >
                    Clear
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
