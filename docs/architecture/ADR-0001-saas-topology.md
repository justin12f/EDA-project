# ADR-0001: SaaS service topology

**Status:** Proposed
**Date:** 2026-08-03
**Deciders:** Project owner

## Context

The repository today is a **library plus a CLI**. `main.py` builds a `ContextCreatorAgent`, which drives a Groq-hosted model through a three-tool loop (`meta_data_context` → `data_reader_tool` → `create_context_agent_tool`) and prints markdown. Everything runs in one process:

- `model_tools/object_registry.py` holds dataframes in a module-level `dict[str, Any]`, keyed by a uuid handle. Handles never expire and are visible to whoever holds the process.
- `agents/master_factory.py` binds one backend (`pandas` | `polars` | `spark`) per session and lazily constructs the domain DI layers.
- `database/postgres_manager.py` writes frames straight to a single Postgres reached through `POSTGRES_*` env vars, using `if_exists="replace"`.
- `docker-compose.yml` runs one `app` container (tty, interactive) plus one `db`.

A frontend now exists: a TanStack Start app extracted from Lovable (see ADR-0005). Its central interaction is *agent proposes a pipeline change → human reviews the diff → accept / reject → the ERD and KPIs update*.

Forces:

- Spark and large polars scans take seconds to minutes. HTTP request/response cannot hold them.
- The engine is CPU- and memory-hungry; the web tier must stay small and elastic independently.
- The engine is deeply Python. The frontend is TypeScript/Node. Neither should be rewritten.
- Single maintainer. Operational surface must stay small enough for one person to run.

## Decision

Adopt a **four-process topology inside one monorepo**:

| Process | Runtime | Responsibility |
|---------|---------|----------------|
| `apps/web` | Node (TanStack Start) | SSR, session cookie, CSRF, BFF. Owns *no* business logic. Proxies to the API with a service token and the resolved tenant context. |
| `services/api` | Python (FastAPI + uvicorn) | REST + SSE. AuthZ, quota gate, audit log, Proposal lifecycle. Wraps `AgentMasterFactory`. Never runs long jobs inline. |
| `services/worker` | Python (arq or Celery on Redis) | Executes runs: read → clean → analyse → persist. Emits run events and usage records. |
| infrastructure | Postgres 15, Redis 7, S3-compatible object storage | State, queue/pubsub, blobs. |

The existing packages (`readers/`, `data_cleaning/`, `analyze_data/`, `statistics/`, `preproccesing/`, `models/`, `evaluation/`, `agents/`, `core/`) are promoted to a single installable package `engine/` consumed by both `services/api` and `services/worker`. Their public contract stays `AgentMasterFactory`.

## Options Considered

### Option A — Keep one Python process, add FastAPI to it

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | Lowest |
| Scalability | Poor — one slow Spark job blocks the event loop and every other tenant |
| Team familiarity | High |

**Pros:** ships fastest; no queue to operate; the `ObjectRegistry` keeps working as-is.
**Cons:** a single tenant's 48M-row scan degrades everyone; no way to scale web separately from compute; process restarts drop every in-flight handle; a worker crash takes the API down with it.

### Option B — API + worker split on a Redis queue (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost | One extra container + Redis |
| Scalability | Good — workers scale horizontally, API stays thin |
| Team familiarity | Medium (arq is small; Celery is well known) |

**Pros:** long jobs never touch the request path; workers can be sized for Spark while the API stays 256MB; run state is durable across restarts; naturally produces the run/event stream the UI already renders.
**Cons:** dataframes can no longer live in a shared in-process dict — requires the DatasetHandle service in ADR-0003; two Python deployables to version together.

### Option C — Serverless functions per step (Lambda / Cloud Run jobs)

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost | Low at idle, spiky under load |
| Scalability | Excellent |
| Team familiarity | Low |

**Pros:** scale to zero; per-step billing granularity that maps neatly onto usage metering.
**Cons:** Spark does not fit the model; cold starts on a 700MB pyspark image are brutal; local development and debugging get much worse for a solo maintainer.

### Option D — Node-only rewrite, drop Python

Rejected without scoring. It discards ~288 Python modules including the entire statistics domain tree and every backend-native implementation. The cost is measured in months and the benefit is one fewer runtime.

## Trade-off Analysis

The decisive constraint is **job duration variance**. A CSV of 480K rows profiles in under a second; a Spark job over 48M rows does not. Any design that serves both from the request path is wrong, which eliminates Option A. Option C solves the same problem but pays for it in operational unfamiliarity and a Spark story that does not work. Option B buys the isolation at the price of one Redis and one extra image — a price a solo maintainer can pay.

The second constraint is **the frontend already assumes asynchrony**: `AGENT ACTIVE`, `AWAITING REVIEW`, `est. 1.2s`, streaming diffs. Option B produces exactly that event model for free; Option A would have to fake it.

Keeping `apps/web` logic-free matters more than it looks. It means the API is the only place authorization and metering are enforced, so there is one place to audit rather than two.

## Consequences

**Easier**
- Scaling compute without scaling the web tier, and vice versa.
- Surviving deploys: runs are rows in Postgres, not objects in RAM.
- Charging accurately: the worker is the single place that knows how long work took.
- Adding a second agent or a new backend — both are engine-level concerns the API never sees.

**Harder**
- Passing dataframes between steps. The module-level `_data_store` in `model_tools/object_registry.py` is now a correctness bug, not just a smell: two workers do not share it. ADR-0003 replaces it.
- Local development requires four processes. Mitigated with a single `docker compose up` profile and a `make dev` target.
- Schema and API contract must move together across two Python deployables.

**To revisit**
- Whether Spark stays in-process in the worker or moves to a dedicated Spark cluster. Deferred until a real tenant needs it; today `pandas`/`polars` cover the tested paths.
- Whether `apps/web` should call the API directly from the browser once auth moves to tokens. Currently the BFF proxy is preferred so no service token reaches the client.

## Action Items

1. [ ] Create the monorepo layout: `apps/web`, `services/api`, `services/worker`, `engine/`, `infra/`.
2. [ ] Move existing top-level Python packages under `engine/` with a `pyproject.toml`; keep import paths stable via package re-export so `agents.master_factory` still resolves.
3. [ ] Stand up `services/api` with health, OpenAPI, and a single `POST /v1/runs` that enqueues.
4. [ ] Stand up `services/worker` with arq, Redis, and one job type that executes a read.
5. [ ] Rewrite `docker-compose.yml` for web + api + worker + db + redis + minio.
6. [ ] Delete the interactive `tty`/`stdin_open` app service once the API replaces the CLI entry point.
