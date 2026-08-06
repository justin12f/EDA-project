import { useState } from "react";

import { apiPost } from "../../lib/api/client";
import type { ImpactReport, Proposal, ProposalDecision } from "../../lib/api/types";

const WORD_BREAK = /_/g;

function humanize(value: string): string {
  return value.replace(WORD_BREAK, " ");
}

function stepName(step: Record<string, unknown>): string {
  const [name] = Object.keys(step);
  return humanize(name ?? "step");
}

const KIND_LABEL: Record<string, string> = {
  cleaning_pipeline: "Cleaning pipeline proposed",
  pipeline_patch: "Sentinel proposed a fix",
  plan_change: "Plan change proposed",
  member_role_change: "Role change proposed",
  entity_mapping: "Glossary entry proposed",
};

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
  // Held locally because a plan_change's checkout_url is only ever in this
  // one response — the persisted proposal row that a later refetch returns
  // has no column for it.
  const [result, setResult] = useState<ProposalDecision | null>(null);

  async function decide(decision: "accept" | "reject") {
    setBusy(decision);
    setError(null);
    try {
      const decided = await apiPost<ProposalDecision>(`/v1/proposals/${proposal.id}/decide`, {
        decision,
      });
      setResult(decided);
      onDecided(proposal.id, decided);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save your decision");
    } finally {
      setBusy(null);
    }
  }

  const pending = proposal.status === "awaiting_review";
  const applied = result?.status === "applied" || proposal.status === "applied";

  return (
    <div className="ai-ring w-full max-w-lg rounded-xl border border-ai-soft bg-card p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="label-eyebrow text-ai">{KIND_LABEL[proposal.kind] ?? "Change proposed"}</span>
        <StatusPill status={proposal.status} />
      </div>

      <ProposalBody proposal={proposal} />

      <p className="mb-3 text-[13px] text-muted-foreground">{proposal.rationale}</p>

      {proposal.impact && <ImpactPanel impact={proposal.impact} />}

      {error && <p className="mb-3 text-[13px] text-destructive">{error}</p>}

      {applied && result?.checkout_url && (
        <a
          href={result.checkout_url}
          target="_blank"
          rel="noopener noreferrer"
          className="mb-3 flex h-8 items-center justify-center rounded-lg bg-primary text-[13px] font-medium text-primary-foreground transition hover:brightness-110"
        >
          Complete checkout →
        </a>
      )}
      {applied && result?.note && (
        <p className="mb-3 text-[13px] text-muted-foreground">{result.note}</p>
      )}

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
      ) : applied ? (
        <p className="text-[13px] text-success">Applied — this proposal already ran.</p>
      ) : (
        <p className="text-[13px] text-muted-foreground">Rejected.</p>
      )}
    </div>
  );
}

function ProposalBody({ proposal }: { proposal: Proposal }) {
  if (proposal.kind === "cleaning_pipeline" || proposal.kind === "pipeline_patch") {
    const steps = (proposal.spec.steps as Array<Record<string, unknown>> | undefined) ?? [];
    return (
      <ul className="mb-3 space-y-1">
        {steps.map((step, index) => (
          <li key={index} className="flex items-center gap-2 text-[13px] text-foreground">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-ai" />
            <span className="capitalize">{stepName(step)}</span>
          </li>
        ))}
      </ul>
    );
  }
  if (proposal.kind === "plan_change") {
    const planCode = String(proposal.spec.plan_code ?? "");
    return (
      <p className="mb-3 text-[13px] text-foreground">
        Switch to the <span className="font-medium capitalize">{planCode}</span> plan.
      </p>
    );
  }
  if (proposal.kind === "member_role_change") {
    const role = String(proposal.spec.role ?? "");
    const previous = String(proposal.spec.previous_role ?? "");
    return (
      <p className="mb-3 text-[13px] text-foreground">
        Change role from <span className="font-medium capitalize">{previous}</span> to{" "}
        <span className="font-medium capitalize">{role}</span>.
      </p>
    );
  }
  if (proposal.kind === "entity_mapping") {
    const name = String(proposal.spec.canonical_name ?? "");
    const members =
      (proposal.spec.members as Array<{ source_id: string; column: string }> | undefined) ?? [];
    return (
      <div className="mb-3">
        <p className="mb-1.5 text-[13px] text-foreground">
          These columns are the same concept: <span className="font-medium">{name}</span>
        </p>
        <ul className="space-y-1">
          {members.map((member, index) => (
            <li key={index} className="flex items-center gap-2 text-[13px] text-foreground">
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-ai" />
              <span className="font-mono text-[12px]">{member.column}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  return null;
}

function formatMetricValue(metric: string, value: number): string {
  return metric === "match_rate" ? `${Math.round(value * 100)}%` : value.toFixed(2);
}

/**
 * What else this change touches (ADR-0010) — already computed by the time
 * this card renders, never fetched separately. Only `cleaning_pipeline` /
 * `pipeline_patch` proposals ever have one; every other kind's
 * `proposal.impact` is null and this never mounts.
 */
function ImpactPanel({ impact }: { impact: ImpactReport }) {
  const truncated = impact.dependents_total > impact.dependents_checked;
  const coverage = truncated
    ? `${impact.dependents_checked} of ${impact.dependents_total} dependents checked`
    : `${impact.dependents_checked} dependent${impact.dependents_checked === 1 ? "" : "s"} checked`;

  return (
    <div className="mb-3 rounded-lg bg-secondary px-3 py-2">
      <p className="mb-1 text-[11px] font-medium text-muted-foreground">Impact — {coverage}</p>
      <p className="text-[13px] text-foreground">{impact.summary}</p>
      {impact.findings.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {impact.findings.map((finding, index) => (
            <li key={index} className="text-[12px] text-muted-foreground">
              <span className="capitalize">{humanize(finding.artifact_kind)}</span>{" "}
              {humanize(finding.metric)}: {formatMetricValue(finding.metric, finding.before)} →{" "}
              <span
                className={`font-medium ${finding.delta_pct < 0 ? "text-destructive" : "text-success"}`}
              >
                {formatMetricValue(finding.metric, finding.after)}
              </span>
            </li>
          ))}
        </ul>
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
      {humanize(status)}
    </span>
  );
}
