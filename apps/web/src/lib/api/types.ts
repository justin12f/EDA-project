export interface Source {
  id: string;
  name: string;
  kind: "postgres" | "mysql" | "csv" | "json" | "parquet";
  status: "idle" | "syncing" | "live" | "error";
  row_count: number | null;
  byte_size: number | null;
  created_at: string;
}

export interface Proposal {
  id: string;
  run_id: string;
  thread_id: string;
  author_agent: string;
  kind: string;
  status: "draft" | "awaiting_review" | "accepted" | "rejected" | "applied" | "failed";
  spec: { rid: string; steps: Array<Record<string, unknown>> };
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
}

export type TranscriptItem =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string }
  | { kind: "tool_call"; id: string; name: string; args: Record<string, unknown> }
  | { kind: "tool_result"; id: string; name: string; ok: boolean }
  | { kind: "error"; id: string; text: string };
