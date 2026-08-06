export interface Source {
  id: string;
  name: string;
  kind: "postgres" | "mysql" | "csv" | "json" | "parquet";
  status: "idle" | "syncing" | "live" | "error";
  row_count: number | null;
  byte_size: number | null;
  created_at: string;
}

// `spec` varies by kind: {rid, steps} for cleaning_pipeline and pipeline_patch
// (the Sentinel's own revision of one — ADR-0008), {plan_code, price_cents}
// for plan_change, {user_id, role, previous_role} for member_role_change,
// {canonical_name, canonical_type, members, reconciliation_rule} for
// entity_mapping (ADR-0009). Narrow it at the read site, not here.
export interface Proposal {
  id: string;
  run_id: string;
  thread_id: string;
  author_agent: string;
  kind:
    | "cleaning_pipeline"
    | "pipeline_patch"
    | "plan_change"
    | "member_role_change"
    | "entity_mapping"
    | (string & {});
  status: "draft" | "awaiting_review" | "accepted" | "rejected" | "applied" | "failed";
  spec: Record<string, unknown>;
  rationale: string;
  applied_run_id: string | null;
  created_at: string;
}

// ADR-0008: a source's scheduled drift scan, and what it has found.
export interface Schedule {
  id: string | null;
  source_id: string;
  enabled: boolean;
  auto_apply_enabled: boolean;
  cron: string | null;
  last_run_at: string | null;
  next_run_at: string | null;
}

export interface SchemaChange {
  kind: "added" | "removed" | "renamed" | "type_changed";
  column: string;
  detail: string;
  previous_column?: string | null;
}

export interface NullRateShift {
  column: string;
  previous: number;
  current: number;
  delta: number;
}

export interface DriftEvent {
  id: string;
  kind: "schema_change" | "statistical_shift" | (string & {});
  severity: number;
  details: { schema_changes?: SchemaChange[]; null_rate_shifts?: NullRateShift[] };
  status: "detected" | "diagnosing" | "proposed" | "resolved" | "dismissed";
  proposal_id: string | null;
  occurred_at: string;
}

export interface ProposalDecision {
  id: string;
  status: string;
  applied_run_id?: string;
  result?: { rid: string; row_count: number; schema: Record<string, string> };
  report?: string;
  // plan_change
  plan_code?: string;
  checkout_url?: string | null;
  note?: string | null;
  // member_role_change
  user_id?: string;
  role?: string;
  // entity_mapping
  entity?: {
    id: string;
    name: string;
    entity_type: string;
    members: Array<{ source_id: string; column: string }>;
  };
}

// ADR-0009: the read-only trust API's credential. `key` only ever appears in
// the response to the create call that minted it — every other read of this
// type (list) omits it entirely.
export interface ApiKey {
  id: string;
  name: string;
  key_prefix: string;
  scope: "read:glossary" | "read:certification";
  created_at: string;
  revoked_at: string | null;
}

export interface ApiKeyCreated extends ApiKey {
  key: string;
}

// ADR-0009 §3: composed fresh from DriftEvent/Proposal state on every read,
// never stored — see services/api/src/lumen_api/certification.py.
export interface Certification {
  certified: boolean;
  open_drift_count: number;
  pending_pipeline_review: boolean;
  unresolved_entity_columns: string[];
  last_checked_at: string | null;
  checked_by: "scheduled_tick" | "manual_profile" | null;
}

export interface UsageStatus {
  metric: string;
  used: number;
  limit: number | null;
  ratio: number | null;
  decision: "allow" | "warn" | "deny";
}

export interface UsageSummary {
  plan: { code: string; name: string; price_cents: number };
  usage: UsageStatus[];
}

export type TranscriptItem =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string }
  | { kind: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { kind: "tool_result"; id: string; name: string; ok: boolean }
  | { kind: "error"; id: string; text: string };
