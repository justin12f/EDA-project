# ADR-0004: Usage metering, quotas, billing and the admin agent

**Status:** Proposed
**Date:** 2026-08-03
**Deciders:** Project owner

## Context

Nothing in the repository measures anything. The Groq client is constructed at import time from an env-var substring scan, calls are unbounded, and no record of a call survives the process. Meanwhile the product's unit economics are dominated by exactly the things nobody counts: LLM tokens, worker CPU-seconds, rows scanned, and bytes stored.

The frontend already sells a plan (`PRO` badge, plan distribution across `free / pro / team / ent / trial`) and displays a live cost signal (`1,840 tokens · gpt-4o`). It shows a billing surface that has no backing.

Two failure modes matter and they pull in opposite directions:

- **Runaway spend.** An agent loop with no cap, a Spark job over 48M rows, or one abusive free-tier account can produce a bill larger than the subscription. This is a hard-stop problem.
- **Broken work.** Killing a run mid-flight because a counter crossed a line loses the customer's work and their trust. This is a graceful-degradation problem.

## Decision

### 1. UsageRecord is written in the same transaction as the work

```
UsageRecord
  id, org_id, run_id, agent, metric, quantity, unit_cost_micros, cost_micros, occurred_at, metadata jsonb
```

Metrics: `llm_input_tokens`, `llm_output_tokens`, `compute_seconds`, `rows_scanned`, `storage_bytes_day`, `agent_run`.

The write is transactional with the work record, not a side effect of it. A run that completed and was not billed is a bug the schema makes impossible: the worker commits the run result and its usage rows together, or neither.

### 2. Quota is enforced at admission, not at the end

Every entry point that can spend money — enqueueing a run, an agent's next LLM call, a materialisation — passes through one `QuotaGate.check(org_id, metric, estimated_quantity)` before starting. The gate reads a per-org counter cached in Redis, backed by the `usage_records` aggregate, and returns `allow | warn | deny` against the plan's limits.

Three thresholds per metric per plan:

| Level | Behaviour |
|-------|-----------|
| soft (80%) | `warn` — banner in the UI, event on the run, work proceeds |
| hard (100%) | `deny` for *new* work; in-flight runs finish |
| kill (120%) | in-flight runs are cancelled at their next checkpoint |

In-flight runs finishing past the hard limit is deliberate: it bounds the overshoot to one run's worth of spend, which is a far better trade than destroying a customer's half-finished pipeline. The 120% kill switch bounds the pathological case.

### 3. The agent loop is budgeted, not just the account

Each `Run` carries a token budget and a wall-clock deadline derived from the plan. The supervisor checks remaining budget before every model call and degrades in order: drop to the `fast` model tier, then reduce context, then stop and emit a `budget_exhausted` event with partial results retained. A run never silently truncates.

### 4. Billing is Stripe, and Stripe is not the source of truth for usage

Stripe holds subscriptions, plans, invoices and payment methods. Usage aggregates are pushed to Stripe metered billing on a schedule. Our `usage_records` table remains authoritative; Stripe is a downstream consumer. A Stripe outage delays invoicing, it does not stop the product or corrupt the ledger.

Plan definitions live in the database (`plans` table — the ERD already models `plan_code`, `plan_name`, `price_cents`), not in code, so limits can be changed without a deploy.

### 5. The AdminAgent — agents that administer the use of the app

A specialist agent (ADR-0003) whose tools are read-mostly and whose scope is the *account*, not the data:

| Tool | Access |
|------|--------|
| `get_usage(period, group_by)` | read |
| `get_quota_status()` | read |
| `explain_cost(run_id)` | read |
| `list_members()` | read |
| `recommend_plan()` | read |
| `suggest_cost_optimisation()` | read — e.g. "this source is scanned daily at 48M rows; cache it as Parquet" |
| `propose_plan_change(plan_code)` | **Proposal** — owner must accept |
| `propose_member_role_change(user, role)` | **Proposal** — owner must accept |

Every write goes through the same Proposal/accept path as data changes. An agent may recommend an upgrade; only a human with the `owner` role buys one. Payment details are never handled by an agent — a plan-change proposal, once accepted, redirects the owner to Stripe Checkout.

## Options Considered

### Option A — No metering; flat-rate plans, absorb the variance

| Dimension | Assessment |
|-----------|------------|
| Complexity | Lowest |
| Cost risk | Unbounded |
| Time to ship | Fastest |

**Pros:** ships now; simplest possible pricing page; no counters to keep correct.
**Cons:** a single tenant running Spark over 48M rows on a $29 plan is a loss with no ceiling; there is no data to price on later; abuse has no technical answer, only an email.

### Option B — Meter into a time-series/analytics store (ClickHouse, Timescale)

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost | Another database to run |
| Accuracy | Excellent at high cardinality |

**Pros:** built for this shape of data; cheap aggregates over millions of events; good foundation for customer-facing analytics.
**Cons:** a second datastore for one maintainer; usage rows can no longer be written in the same transaction as the work, which reopens the "completed but unbilled" gap; overkill at current volume.

### Option C — Meter into Postgres, cache counters in Redis, push aggregates to Stripe (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost | None beyond Redis, already present per ADR-0001 |
| Accuracy | Transactional; exact |

**Pros:** transactional with the work; one datastore; RLS gives per-org usage queries for free; Redis makes the admission check sub-millisecond; a monthly rollup table keeps the hot path small.
**Cons:** raw `usage_records` grows quickly and needs partitioning plus a retention policy; Redis counters can drift from Postgres and require periodic reconciliation.

### Option D — Delegate metering to a vendor (Orb, Lago, Metronome)

**Pros:** correct billing is genuinely hard; these products solve it properly.
**Cons:** a per-event cost and an external dependency in the hot path before there is a single paying customer. Revisit at real revenue.

## Trade-off Analysis

The decisive property is **transactionality**. Usage that is recorded separately from work will, at some point, disagree with it — and the disagreement is discovered during a billing dispute. Option C makes the two atomic, which is worth more than the query performance Option B offers at a volume that does not yet exist. Redis drift is acceptable because Redis is only the admission cache, never the ledger; a nightly reconciliation against Postgres corrects it.

Option A deserves a fair hearing: metering before product-market fit is a classic premature investment. It is rejected because this product's marginal cost per request is high and unbounded — that is the specific case where flat-rate absorption fails. The `UsageRecord` write is a handful of lines at each call site; the counters can stay invisible in the UI until pricing is ready.

The soft/hard/kill ladder rather than a single limit is the direct answer to the two opposing failure modes: a single hard limit either overshoots (too high) or destroys work (too low).

## Consequences

**Easier**
- Pricing on evidence — "p95 tenant spends X" is a query.
- Answering "why is my bill this size" — `explain_cost(run_id)` walks the ledger.
- Capping abuse technically rather than contractually.
- Shipping a customer-facing usage dashboard: the data model is already per-org and per-metric.

**Harder**
- Every spend site must call the gate and write a record. Enforced by making the LLM client and the job runner the only paths that can spend, and putting the calls inside them.
- `usage_records` needs monthly partitioning and a retention window.
- Plan limits become configuration with real consequences, and need their own tests.

**To revisit**
- Moving to Option D once billing complexity (proration, credits, annual plans, overage invoicing) exceeds what a rollup job should own.
- Partitioning or archiving strategy once `usage_records` passes ~10⁸ rows.

## Action Items

1. [ ] Create `plans`, `subscriptions`, `usage_records` (monthly partitions), `usage_rollups` with RLS.
2. [ ] Implement `QuotaGate` with Redis counters and Postgres reconciliation.
3. [ ] Make the `LLMProvider` wrapper the only path to a model, and have it write `llm_*_tokens` records with the real model id.
4. [ ] Make the worker's job runner the only path to compute, and have it write `compute_seconds` and `rows_scanned`.
5. [ ] Add per-run token budget and deadline; implement tier degradation and `budget_exhausted`.
6. [ ] Integrate Stripe: Checkout for plan changes, webhooks for subscription state, scheduled push of metered aggregates.
7. [ ] Implement `AdminAgent` with the read-mostly toolset; route both write tools through Proposal.
8. [ ] Build the usage panel in the frontend against `GET /v1/usage`.
