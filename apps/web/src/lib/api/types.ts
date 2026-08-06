export interface Source {
  id: string;
  name: string;
  kind: "postgres" | "mysql" | "csv" | "json" | "parquet";
  status: "idle" | "syncing" | "live" | "error";
  row_count: number | null;
  byte_size: number | null;
  created_at: string;
}

// `spec` varies by kind: {rid, steps} for cleaning_pipeline, {plan_code,
// price_cents} for plan_change, {user_id, role, previous_role} for
// member_role_change. Narrow it at the read site, not here.
export interface Proposal {
  id: string;
  run_id: string;
  thread_id: string;
  author_agent: string;
  kind: "cleaning_pipeline" | "plan_change" | "member_role_change" | (string & {});
  status: "draft" | "awaiting_review" | "accepted" | "rejected" | "applied" | "failed";
  spec: Record<string, unknown>;
  rationale: string;
  applied_run_id: string | null;
  created_at: string;
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
