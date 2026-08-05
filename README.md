# Lumen — agentic data platform

Connect a data source, let an agent audit it, review every change before it lands.

The agent profiles your data, proposes a cleaning pipeline as an executable spec,
and waits. Nothing touches your data until a human accepts the proposal. What it
learns is embedded and stored, so the second visit to a source starts from what it
concluded on the first.

---

## Setting up your API keys

**You do not need one to run this.** With no key configured, agents use a local
deterministic provider that profiles real data and proposes real pipelines — a
key changes the wording of the reasoning, not the shape of the flow.

When you are ready to add one:

1. `cp .env.example .env` if you have not already.
2. Paste your key into `.env`:

   ```dotenv
   ANTHROPIC_API_KEY=sk-ant-...        # console.anthropic.com → API keys
   GROQ_API_KEY=gsk_...                # console.groq.com → API keys
   ```

3. Confirm the app sees them:

   ```bash
   make check-config
   ```

   ```
   llm mode          auto -> anthropic
   anthropic key     set
   groq key          set
   ```

   `auto` picks Anthropic when its key is present, Groq when only that is, and the
   local provider when neither is. Force one with `LLM_MODE=anthropic|groq|mock`.
   An explicit mode never silently degrades — it fails loudly if its key is missing.

4. `GET /v1/config` reports the same thing from a running server, and never
   echoes a secret.

Which model does what — change these without a deploy:

| Setting | Default | Used for |
|---------|---------|----------|
| `MODEL_REASONING` | `claude-opus-5` | supervisor: planning a pipeline, judging a schema |
| `MODEL_SPECIALIST` | `claude-sonnet-5` | the context, cleaning and analysis agents |
| `MODEL_FAST` | `qwen/qwen3.6-27b` | Groq — cheap classification and summarisation |

Guard rails, per run: `AGENT_MAX_ITERATIONS`, `AGENT_DEADLINE_SECONDS`,
`AGENT_MAX_TOTAL_TOKENS`. A run that hits one stops cleanly and keeps its partial
results; it never truncates silently.

---

## Setting up Supabase

Supabase is required — it holds the database, authentication and file storage.

### Hosted project

1. Create a project at [supabase.com](https://supabase.com).
2. **Settings → API** gives you `SUPABASE_URL`, the `anon` key and the
   `service_role` key. **Settings → API → JWT Settings** gives the JWT secret.
3. **Settings → Database → Connection string → URI** gives `DATABASE_URL`. Insert
   `+asyncpg` after `postgresql`, and use **port 5432** — the transaction pooler
   on 6543 does not carry the `SET LOCAL` statements this app's row-level
   security depends on.
4. Fill the Supabase block in `.env`.
5. Apply the schema:

   ```bash
   supabase link --project-ref YOUR-PROJECT-REF
   supabase db push
   ```

### Local project

```bash
supabase start          # prints every key you need
supabase db push
```

The service role key is a **server-only** credential. Only `SUPABASE_ANON_KEY`
belongs in anything prefixed `VITE_`, because that is compiled into the browser
bundle.

---

## Running it

```bash
make setup      # install engine, api and web dependencies
make dev        # supabase + redis
make migrate    # apply supabase/migrations
```

then, in three terminals:

```bash
make api-dev      # http://localhost:8000/docs
make worker-dev
make web-dev      # http://localhost:3000
```

`make test` runs every suite. All of it passes with no credentials configured.

---

## How it fits together

```
browser ── apps/web ── services/api ── Redis ── services/worker
   TanStack Start      FastAPI                    arq
   Supabase Auth       JWT verify, RLS            pipelines, embeddings
        │                   │                          │
        └───────────── Supabase ────────────────────────┘
                Postgres + pgvector + Storage
```

- **`engine/`** — the analytics engine as one `lumen` package: readers, cleaning
  steps, analyzers and eleven statistics domains, each with pandas, polars and
  Spark implementations behind one factory. Knows nothing about HTTP or tenants.
- **`services/api`** — HTTP. Verifies the Supabase JWT, opens every tenant
  transaction as `authenticated` so RLS applies, enqueues work, streams agent
  events over SSE.
- **`services/worker`** — runs pipelines and agent loops off the request path,
  because a scan over 48M rows does not belong in an HTTP handler.
- **`apps/web`** — the product surface.
- **`supabase/migrations`** — every table, policy and function, versioned.

### Two ideas the design turns on

**The agent's output is the engine's input.** A proposal's `spec` is exactly the
JSON `PipelineBuilder.build()` already consumes:

```json
[ { "drop_nulls":      { "columns": ["country_code"] } },
  { "drop_duplicates": { "columns": ["email_hash"], "keep": "last" } } ]
```

No generated code is ever executed. A hallucinated step fails validation against
the engine's factories before a human ever sees it.

**Isolation lives in the database.** Every table carrying `org_id` has row-level
security delegating to one `is_org_member()` function. A forgotten `WHERE` clause
— in hand-written SQL or in something a model produced — returns nothing rather
than another tenant's data. A test enumerates the catalogue and fails the build
if a table appears without a policy.

---

## Documentation

| Document | Subject |
|----------|---------|
| [docs/architecture/](docs/architecture/README.md) | ADRs — topology, tenancy, agent orchestration, metering |
| [ADR-0006](docs/architecture/ADR-0006-supabase-and-vector-context.md) | Supabase and the pgvector context store |
| [design-system.md](docs/architecture/design-system.md) | The visual language — read before writing any UI |
| [docs/superpowers/plans/](docs/superpowers/plans/) | Implementation plans, task by task |
