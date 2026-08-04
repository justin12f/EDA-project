# ADR-0003: Agent orchestration, the Proposal domain, and dataset handles

**Status:** Proposed
**Date:** 2026-08-03
**Deciders:** Project owner

## Context

The agent layer is one third built. `agents/context_creator.py` runs a hand-rolled OpenAI-compatible tool loop against Groq (`qwen/qwen3.6-27b`) with three tools, and returns markdown. `agents/postgres_admin_agent.py` is a plain class that writes a frame to Postgres — it is not reachable by any model. `agents/master_factory.py` can construct cleaning, statistics, encoder, model and evaluation layers that **no agent currently uses**.

Concrete defects in the existing loop:

- `context_creator.py` appends the tool result **twice** for `data_reader_tool` (once inside the `elif`, once in the shared append after it), producing a duplicated `tool` message and an id mismatch risk.
- `MODEL_NAME = "qwen/qwen3.6-27b"` is assigned via a walrus-style double assignment and hardcoded next to `self.client = client_groq`; there is no provider abstraction.
- `while True:` has no iteration cap, no timeout, and no token budget. A model that keeps calling tools loops forever.
- Tool failures raise (`raise RuntimeError(f"Error reading file: {e}")`) instead of being fed back to the model as a tool result, so one bad path kills the run instead of letting the model recover.
- `_serialize_reader_output` is dead code with a bare `except:`.

The frontend fixes the interaction model: an agent **proposes** (`AWAITING REVIEW`, `+ Drop NULLs · country_code`, `Affects ~15,420 rows · est. 1.2s`) and a human **accepts or rejects**. Applied changes then animate through the ERD (`altered` badge) and the KPI tiles (null rate 3.2% → 0.0%).

The phrase driving this work — *"finish the AI agent flow that administers the use of the app"* — has two halves, and both are in scope:
- agents that **operate** the product on the user's behalf (profile, clean, analyse, persist);
- agents and rules that **govern** that operation (budget, quota, approval, audit) — covered jointly here and in ADR-0004.

## Decision

### 1. A supervisor with typed specialist agents

```
SupervisorAgent
├── ContextAgent      profile a source, pick a backend, summarise schema   (exists, hardened)
├── CleaningAgent     propose an ordered cleaning pipeline                 (new)
├── AnalysisAgent     run statistics / EDA, build dashboard widgets        (new)
├── PersistenceAgent  materialise a cleaned frame into the tenant schema   (wrap existing)
└── AdminAgent        answer usage, cost, quota and plan questions         (new — see ADR-0004)
```

The supervisor owns turn-taking and budget. Specialists own tools. All of them share one `ToolRegistry` built from `AgentMasterFactory`, so a tool is a thin typed wrapper over an engine capability — never a reimplementation of it.

### 2. Proposal is a first-class persisted entity

No agent mutates tenant state directly. A specialist emits a `Proposal`:

```
Proposal
  id, org_id, run_id, thread_id, author_agent
  kind:      cleaning_pipeline | schema_change | materialisation | model_fit
  status:    draft | awaiting_review | accepted | rejected | applied | failed
  spec:      jsonb   -- the executable plan, engine-shaped, not prose
  rationale: text    -- why, shown in the UI
  estimate:  jsonb   -- {affected_rows, est_seconds, est_cost_cents}
  decided_by, decided_at
```

`spec` for `kind = cleaning_pipeline` is exactly the list `PipelineBuilder.build()` already consumes:

```json
[ { "drop_nulls":     { "columns": ["country_code"] } },
  { "drop_duplicates":{ "columns": ["email_hash"], "keep": "last" } } ]
```

That is the key design property: **the agent's output is the engine's input**. No translation layer, no generated Python, no `exec`. An agent that proposes an unknown step fails validation at `DataCleaningStepFactory` before any human sees it.

Accepting a proposal enqueues a `Run`. Rejecting records the decision and feeds it back into the thread so the agent can revise.

### 3. DatasetHandle replaces the in-process ObjectRegistry

`model_tools/object_registry.py` — a module-level `dict` — is not merely untidy under ADR-0001, it is incorrect: the API process that creates a handle and the worker process that consumes it are different processes. It is also unbounded (no eviction) and untenanted (any caller with a uuid gets any frame).

Replacement: a `DatasetHandle` service.

```
DatasetHandle
  rid          -- opaque id, unchanged shape so tool signatures survive
  org_id       -- authorisation
  uri          -- s3://org/<org_id>/datasets/<rid>.parquet
  backend      -- pandas | polars | spark
  schema       -- jsonb column/type map
  row_count, byte_size
  created_at, expires_at
```

Materialisation is Parquet in object storage; metadata rows live in Postgres under RLS. `resolve(rid, org_id)` returns a backend-native frame via the existing `ReadersInyeccionDependency` — polars gets `scan_parquet` (lazy), spark gets `spark.read.parquet`, pandas gets `read_parquet`. Handles carry a TTL and are swept.

This keeps every tool signature (`data_reader_tool(file, backend) -> str`) intact while making handles durable, isolated, and inspectable.

### 4. Provider-agnostic LLM client with Claude as the default

`api/groq/qwen_3_6.py` and `api/huggin_face/qwen_3_6.py` both scan `os.environ` for a substring (`"API_KEY_groq" in key`) and print to stdout on failure. Both are replaced by one `LLMProvider` interface with explicit settings.

Default model tier for reasoning-heavy work (planning a pipeline, judging a schema): **Claude** — `claude-opus-5` for the supervisor and `claude-sonnet-5` for specialists, using the Anthropic Messages API with native tool use. Groq/Qwen is retained as a `fast` tier for cheap classification and summarisation. Model choice per agent is configuration, not code, and is recorded on every `UsageRecord`.

### 5. Bounded, observable, resumable loops

Every agent loop gets: a max-iteration cap, a wall-clock deadline, a token budget checked against the org's remaining quota before each call, tool errors returned to the model as tool results rather than raised, and one `AgentEvent` appended per step. `AgentEvent` rows are the SSE stream the UI renders — `thinking`, `tool_call`, `tool_result`, `proposal`, `error`, `done`.

## Options Considered

### Option A — Keep the hand-rolled loop, extend it in place

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | None |
| Control | Total |
| Team familiarity | High — the code already exists |

**Pros:** no new dependency; the existing loop is 40 lines and fully understood; provider-agnostic by construction since it speaks the OpenAI schema.
**Cons:** every one of retries, budgets, streaming, checkpointing and multi-agent handoff must be written and tested by hand.

### Option B — LangGraph

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium-high |
| Cost | Dependency weight; `langchain_core` is already a dependency |
| Control | Good; explicit state machine |
| Team familiarity | Partial — `@tool` decorators already in use |

**Pros:** supervisor/specialist topology is its canonical example; built-in checkpointing maps onto durable runs; the `@tool`-decorated functions in `model_tools/` port directly.
**Cons:** a large abstraction to debug through; version churn; its persistence layer would duplicate the `Run`/`AgentEvent` tables this ADR needs anyway.

### Option C — A thin in-house orchestrator over the existing tool loop (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost | None beyond code |
| Control | Total |
| Team familiarity | High |

**Pros:** the loop stays ~150 readable lines; run state lives in the same Postgres as everything else, so one query answers "what did this agent do and what did it cost"; no framework upgrade can break a customer's run; swapping providers is one interface.
**Cons:** hand-written retry/backoff, streaming, and handoff logic; no community recipes to copy.

## Trade-off Analysis

The honest case for Option B is that supervisor topologies and checkpointing are exactly what it does. The case against it is specific rather than ideological: this system must record, per step, the tenant, the token count, the cost, and the human decision — because ADR-0004 bills on it and ADR-0002 isolates on it. That ledger has to exist as first-class rows regardless. Once it does, the framework's own persistence is a second source of truth for the same facts, and reconciling two ledgers is worse than writing one loop.

Option A is rejected only in its "leave it as is" form; Option C *is* Option A plus the guardrails, which is why the existing `context_creator.py` is hardened rather than discarded.

The Proposal design carries most of the safety weight. Because `spec` is engine-shaped JSON validated by the existing factories, the blast radius of a hallucinating model is "the proposal fails validation" rather than "arbitrary code ran against production data". This is why no code-generation path (`exec`, generated pandas) is on the table.

## Consequences

**Easier**
- Auditing: `SELECT * FROM agent_events WHERE run_id = …` is the complete story of a run, including which human approved what.
- Adding a specialist: register tools, add a prompt, add a proposal kind. No orchestrator surgery.
- Streaming to the UI: `AgentEvent` rows are already the shape the panels render.
- Reproducibility: a `Run` can be replayed from its `Proposal.spec` with no model in the loop.

**Harder**
- `model_tools/*` must be rewritten against `DatasetHandle` — signatures survive, internals do not.
- Prompts become product surface with their own review and versioning.
- Every tool needs a strict input schema; loose `dict` arguments are no longer acceptable.

**To revisit**
- Adopting LangGraph if the topology grows past ~6 specialists or needs sub-graph parallelism.
- Letting an agent apply low-risk proposals without review, once an accuracy baseline exists. Deliberately not day-one behaviour.

## Action Items

1. [ ] Fix `agents/context_creator.py`: remove the duplicated `tool` message append, add iteration cap + deadline, return tool errors as results, delete `_serialize_reader_output`.
2. [ ] Add `engine/llm/` with `LLMProvider`, an Anthropic implementation (default), and a Groq implementation (fast tier); delete the env-substring scanning in `api/groq/` and `api/huggin_face/`.
3. [ ] Implement `DatasetHandle` (Postgres metadata + Parquet in object storage) and retire `model_tools/object_registry.py`.
4. [ ] Define `Proposal`, `Run`, `AgentEvent` tables and their RLS policies.
5. [ ] Build `ToolRegistry` over `AgentMasterFactory`, exposing: `list_data_sources`, `profile_source`, `read_source`, `propose_cleaning_pipeline`, `run_statistic`, `materialise_dataset`, `get_usage`.
6. [ ] Implement `SupervisorAgent` plus `CleaningAgent`, `AnalysisAgent`, `PersistenceAgent`.
7. [ ] Expose `GET /v1/runs/{id}/events` as SSE and wire the frontend chat panel to it.
8. [ ] Validate every `Proposal.spec` against the engine factories before it reaches `awaiting_review`.
