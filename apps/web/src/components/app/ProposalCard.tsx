import { useState } from "react";

import { apiPost } from "../../lib/api/client";
import type { Proposal, ProposalDecision } from "../../lib/api/types";

const STEP_LABEL = /_/g;

function stepName(step: Record<string, unknown>): string {
  const [name] = Object.keys(step);
  return (name ?? "step").replace(STEP_LABEL, " ");
}

/**
 * A proposal waiting on a person, or the record of what they decided. This is
 * the one place in the product where the agent's output becomes an action —
 * everywhere else it only reads and describes.
 */
export function ProposalCard({
  proposal,
  onDecided,
}: {
  proposal: Proposal;
  onDecided: (id: string, result: ProposalDecision) => void;
}) {
  const [busy, setBusy] = useState<"accept" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function decide(decision: "accept" | "reject") {
    setBusy(decision);
    setError(null);
    try {
      const result = await apiPost<ProposalDecision>(`/v1/proposals/${proposal.id}/decide`, {
        decision,
      });
      onDecided(proposal.id, result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save your decision");
    } finally {
      setBusy(null);
    }
  }

  const pending = proposal.status === "awaiting_review";

  return (
    <div className="ai-ring w-full max-w-lg rounded-xl border border-ai-soft bg-card p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="label-eyebrow text-ai">Cleaning pipeline proposed</span>
        <StatusPill status={proposal.status} />
      </div>

      <ul className="mb-3 space-y-1">
        {proposal.spec.steps.map((step, index) => (
          <li key={index} className="flex items-center gap-2 text-[13px] text-foreground">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-ai" />
            <span className="capitalize">{stepName(step)}</span>
          </li>
        ))}
      </ul>

      <p className="mb-3 text-[13px] text-muted-foreground">{proposal.rationale}</p>

      {error && <p className="mb-3 text-[13px] text-destructive">{error}</p>}

      {pending ? (
        <div className="flex gap-2">
          <button
            onClick={() => decide("accept")}
            disabled={busy !== null}
            className="h-8 flex-1 rounded-lg bg-primary text-[13px] font-medium text-primary-foreground transition hover:brightness-110 disabled:opacity-50"
          >
            {busy === "accept" ? "Applying…" : "Accept"}
          </button>
          <button
            onClick={() => decide("reject")}
            disabled={busy !== null}
            className="h-8 flex-1 rounded-lg border border-input bg-background text-[13px] font-medium text-foreground transition hover:bg-secondary disabled:opacity-50"
          >
            {busy === "reject" ? "…" : "Reject"}
          </button>
        </div>
      ) : proposal.status === "applied" ? (
        <p className="text-[13px] text-success">Applied — this proposal already ran.</p>
      ) : (
        <p className="text-[13px] text-muted-foreground">Rejected.</p>
      )}
    </div>
  );
}

function StatusPill({ status }: { status: Proposal["status"] }) {
  const styles: Record<string, string> = {
    awaiting_review: "bg-ai-tint text-ai",
    applied: "bg-success-tint text-success",
    rejected: "bg-muted text-muted-foreground",
    failed: "bg-destructive-tint text-destructive",
  };
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${styles[status] ?? "bg-muted text-muted-foreground"}`}
    >
      {status.replace(/_/g, " ")}
    </span>
  );
}
