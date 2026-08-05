# Architecture

## Documents

| Doc | Subject |
|-----|---------|
| [factories.md](factories.md) | Existing factory / DI layering of the analytics engine |
| [ADR-0001-saas-topology.md](ADR-0001-saas-topology.md) | Service topology for the SaaS |
| [ADR-0002-multi-tenancy-and-auth.md](ADR-0002-multi-tenancy-and-auth.md) | Tenancy model, identity, data isolation |
| [ADR-0003-agent-orchestration.md](ADR-0003-agent-orchestration.md) | Agent layer, Proposal/Run domain, dataset handles |
| [ADR-0004-usage-metering-and-governance.md](ADR-0004-usage-metering-and-governance.md) | Metering, quotas, cost control, admin agent |
| [ADR-0005-frontend-port.md](ADR-0005-frontend-port.md) | Porting the Lovable frontend off proprietary packages |
| [design-system.md](design-system.md) | Visual language extracted from the Lovable prototype |

## System at a glance

```
                    ┌──────────────────────────────────────────┐
  browser ──────────│ apps/web  ·  TanStack Start (SSR + BFF)   │
                    │  · session cookie, CSRF, rate limit      │
                    │  · streams agent events (SSE)            │
                    └───────────────┬──────────────────────────┘
                                    │  internal HTTP (service token + tenant ctx)
                    ┌───────────────▼──────────────────────────┐
                    │ services/api  ·  FastAPI                 │
                    │  · REST + SSE                            │
                    │  · authz, quota gate, audit              │
                    │  · AgentMasterFactory (existing engine)  │
                    └──────┬─────────────────┬─────────────────┘
                           │ enqueue          │ read/write
                    ┌──────▼──────┐    ┌──────▼──────────────┐
                    │ Redis queue │    │ Postgres (RLS)      │
                    └──────┬──────┘    │  control plane +    │
                           │           │  tenant warehouses  │
                    ┌──────▼──────────┐└─────────────────────┘
                    │ services/worker │
                    │  · runs pipelines (pandas/polars/spark) │
                    │  · emits run events + usage records     │
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │ object storage  │  uploads, artifacts, parquet cache
                    └─────────────────┘
```

## Non-negotiables

1. The analytics engine (`readers/`, `data_cleaning/`, `analyze_data/`, `statistics/`, `models/`, `evaluation/`) stays backend-agnostic and free of web/tenant concerns. The API layer adapts; the engine does not learn about HTTP.
2. Every mutating agent action produces a **Proposal** that a human accepts or rejects. Nothing an agent decides touches tenant data without a recorded decision.
3. Every unit of work that costs money (LLM tokens, compute seconds, rows scanned, storage bytes) emits a **UsageRecord** in the same transaction that records the work.
4. Tenant isolation is enforced in the database (RLS), not only in application code.
