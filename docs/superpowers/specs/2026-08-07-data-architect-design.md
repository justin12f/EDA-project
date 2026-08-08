# The Data Architect — design

**Date:** 2026-08-07
**Status:** Approved for planning
**Implements:** ADR-0024 (to be written alongside)
**Extends:** ADR-0003 (proposals), ADR-0008 (drift), ADR-0009 (canonical entities), ADR-0010 (impact/lineage), ADR-0011 (progressive trust), ADR-0013 (baselines)
**Language note:** written in English to match `docs/architecture/*` and the codebase; the design conversation was in Spanish.

---

## 1. Context

Lumen today never builds a database. `POST /v1/sources` writes raw bytes to Supabase Storage at `org/<org_id>/uploads/<source_id>.csv` and inserts one `data_sources` row — it does not parse the file. A customer's "data" is N metadata rows plus Parquet blobs referenced by `dataset_handles`. `data_sources` reserves `dsn_encrypted` and `table_name` columns and **no code path ever writes them**; `source_kind` includes `postgres` and `mysql` and neither is implemented.

Three consequences motivate this work:

1. **There is nothing to show.** A customer cannot see their own data, and there is no structure to diagram. `canonical_entities` (ADR-0009) is the only cross-source relationship in the system, it is an undirected equivalence class with no direction or cardinality, and it only materialises when embedding clustering exceeds cosine ≥ 0.92 on a worker tick — most workspaces have zero.
2. **Analysis has no substrate a customer can reason about.** Every capability built in ADR-0008 through ADR-0013 operates on in-process DataFrames materialised from Parquet. Correct for compute, invisible to the user.
3. **Customers arrive with databases, not just files.** The product must ingest any format, and must accept a customer connecting their own database(s) to be cleaned and analysed.

This design introduces an agent that **plans, creates, and administers a real relational database per customer**, and makes that database the source of truth for analysis.

---

## 2. Decisions

Ten decisions were taken during design. Each is binding on the plan.

| # | Decision | Rationale |
|---|---|---|
| D1 | **A dedicated Postgres instance for tenant data, outside Supabase** — schema-per-org, with **one Postgres role per org** | Supabase keeps the control plane only. The tenant instance holds no `organizations`, `subscriptions`, or `api_keys` at all, so even a total failure of role isolation cannot reach them — the blast radius is bounded by what the instance contains, not by what a grant forbids. FKs are still genuinely enforced, so the diagram is *read* from `information_schema` rather than guessed |
| D2 | **Conservative 1:1 modelling** — one table per source, plus real types, a justified PK, and declared FKs | The customer recognises their own data; the diagram still gets edges from cross-source FKs; and no `shadow_run` equivalent exists to validate a schema redesign |
| D3 | **SQL is the source of truth; Parquet becomes a compute cache** | The engine's contract is already "give me a DataFrame" (`HandleStore.resolve()`), so `pl.read_database()` satisfies it with zero engine changes |
| D4 | **Staging is immediate; promotion is approved** | Breaks the circularity that a human cannot judge a proposed schema without seeing the data it describes |
| D5 | **FKs are graded** — enforced when containment is total, observed when partial | Degrades honestly; maps to solid/dashed edges; an orphan row becomes a data-quality finding, not a hard failure |
| D6 | **Re-ingest is full replacement**, inside one transaction with deferred constraints | A file is a *snapshot*, not a delta feed; an upsert would silently retain rows the customer deleted at origin. Row history is unnecessary — ADR-0013 already stores history as calibrated baselines and drift events |
| D7 | **Schema evolution is gated by reversibility**, reusing ADR-0008's confidence ladder and ADR-0011's trust streak. **Never `DROP COLUMN`** | Making nothing destructive dissolves ADR-0017 §3's irreversibility ceiling, which would otherwise permanently block autonomy |
| D8 | **Deterministic first; the model is used only for judgment** | Matches ADR-0008 ("detection stays model-free") and ADR-0013 ("classical statistics, not a model call"). Also keeps the keyless `MockProvider` path working |
| D9 | **Ingestion is an adapter layer**, format-agnostic by contract | Any format, and live customer databases, through one interface |
| D10 | **Customer databases: mirror structure immediately, copy data per table on demand** | The diagram appears in seconds with *real* FKs and no bytes copied; avoids dragging a 500-table ERP to look at three |
| D11 | **All of one org's sources land in one database**, so cross-source FKs are enforceable. The agent decides naming and whether to merge or namespace | Postgres enforces FKs across schemas within a database but never across databases. Putting every source of an org in one database is what makes "a customer connects two databases and the FK crosses between them" physically possible rather than decorative |
| D12 | **DuckDB is a read accelerator, not a fourth backend** — used above a row threshold | `Backend` is `Literal["pandas","polars","spark"]`; a fourth value would touch `validate_backend` and every per-backend dispatch in `materialize.py`, `data_cleaning/steps/backends/`, and `statistics/*/backends/`. DuckDB's `postgres` extension returns Arrow, which converts to a polars frame with no copy — so D3's promise that the engine does not change survives. Spark stays available as the next tier and is **not** enabled now |

---

## 3. Architecture

### 3.1 Two instances

Tenant data lives in a **dedicated Postgres instance**, separate from Supabase (D1):

| Instance | Holds | Accessed via |
|---|---|---|
| **Supabase** (existing) | control plane only — `organizations`, `memberships`, `subscriptions`, `api_keys`, `proposals`, `drift_events`, `column_baselines`, `data_contexts`… | `user_session()`, `service_session()` |
| **Tenant Postgres** (new) | customer data only — one schema per org | `tenant_session()` |

The security argument is now structural rather than permissive: the tenant instance contains no control-plane table for an agent to reach, so the blast radius is bounded by what the instance *holds*, not by what a `REVOKE` forbids. `db/session.py` gains a second engine and pool, keyed the same per-event-loop way the existing one is.

Per-org roles are still required — they separate **tenants from each other** inside the tenant instance, which is a different problem from separating tenants from the control plane.

### 3.2 Tenant isolation

This is the security crux: **an agent writes the DDL**.

At org bootstrap, three objects are created **in the tenant instance**:

| Object | Name | Purpose |
|---|---|---|
| Schema | `tenant_<hex>` | The modelled database — what the customer sees and analyses |
| Schema | `tenant_<hex>_raw` | Staging — data lands here on upload, before any schema is approved |
| Role | `tenant_<hex>_role` | Owner of both schemas, **with no grant whatsoever on `public`** |

`<hex>` is the org uuid with dashes stripped (32 chars). Longest identifier is `tenant_<hex>_role` at 44 bytes, within Postgres's 63-byte limit.

Every operation on customer data runs under `SET LOCAL ROLE tenant_<hex>_role`. Isolation is enforced by Postgres grants, not by application logic — the same principle `db/session.py` already states for RLS: *"Isolation is the database's job."* A bug in the Architect cannot read another org or touch the control plane, because the role has no permission to.

A new session helper mirrors the existing ones:

```
tenant_session(org_id)      # SET LOCAL ROLE tenant_<hex>_role
                            # SET LOCAL search_path = tenant_<hex>, tenant_<hex>_raw
```

It joins `user_session()` and `service_session()` in `db/session.py`, and that module's docstring — which enumerates every legitimate `service_session()` call site — must be extended to describe it.

**The two-session rule.** A tenant session **cannot** write `proposals`, `artifact_dependencies`, `drift_events`, or any other control-plane row — not because a grant forbids it but because those tables are in a different database on a different instance. There is no transaction that can span both. Every operation touching both therefore uses two sessions in sequence:

```
async with tenant_session(org_id) as tdb:   # DDL / DML on customer data
    ...
async with user_session(user_id) as db:     # proposals, lineage, drift
    ...
```

This is the same shape `proposals.py` already uses to record an ADR-0011 trust decision in a session separate from the apply, and it carries the same trade-off: a failure between the two leaves one side written and the other not. Ordering is chosen so the recoverable state survives — **the control-plane record is written last**, so a crash leaves an applied schema with a proposal still marked `accepted`, never a record of a schema that does not exist. Reconciliation is a re-run of `discover()` against the tenant schema, which is authoritative.

**Provisioning is lazy and idempotent.** Orgs already exist without these objects, so this cannot live only in the signup path. `ensure_tenant_schema(org_id)` runs at the head of every ingestion job, creates what is missing with `IF NOT EXISTS` and the `DO $$ … EXCEPTION WHEN duplicate_object` pattern every migration in this repo already uses, and is safe to call concurrently.

### 3.3 The `SchemaSpec` contract

The agent **never emits SQL**. It emits a typed structure the engine renders:

```
SchemaSpec
  tables: TableSpec[]
  foreign_keys: ForeignKeySpec[]

TableSpec
  name           str          # sanitised identifier
  source_id      uuid
  source_table   str | None   # origin table name, for DB sources
  columns        ColumnSpec[]
  primary_key    str[] | None
  pk_rationale   str

ColumnSpec
  name           str          # sanitised
  source_column  str          # original name, preserved verbatim
  sql_type       SqlType      # closed enum
  nullable       bool
  deprecated     bool = False # absent at origin but retained (D7)

ForeignKeySpec
  from_table, from_column  str
  to_table,   to_column    str
  containment    float      # 1.0 => enforced, <1.0 => observed
  enforced       bool
  evidence       Evidence[] # declared | structural | semantic | naming
  rationale      str
```

`SqlType` is a closed enum: `text`, `varchar(n)`, `integer`, `bigint`, `numeric(p,s)`, `double precision`, `boolean`, `date`, `timestamp`, `timestamptz`, `uuid`, `jsonb`. Mapping from polars/pandas dtypes is deterministic and table-driven.

`Evidence` ranks the strength of a claim, and `declared` (read from a customer database's own constraints) outranks everything inferred.

### 3.4 DDL rendering — the one injection surface

`render_ddl(spec) -> list[str]` lives in the engine and is pure. It is the **only** place in the system that composes SQL text, and therefore the only place that needs injection review:

- Identifiers are sanitised (lowercase, non-alphanumeric → `_`, collapse runs, strip leading digits, truncate to 63 bytes, de-duplicate collisions with a numeric suffix) and then **validated** against `^[a-z_][a-z0-9_]*$`. A value that fails validation is a bug, not an escape hatch — it raises.
- Types come from the closed enum; they are never interpolated from model output.
- Reserved words are handled by quoting *and* checked against a reserved list so generated names avoid them.

Foreign keys are created `DEFERRABLE INITIALLY IMMEDIATE`, which is what makes D6's `SET CONSTRAINTS ALL DEFERRED` replacement possible.

`PostgresManager.execute_query(query: str)` — the orphan from the pre-SaaS codebase at `engine/src/lumen/database/postgres_manager.py`, imported only by the equally orphaned `PostgresAdminAgent` — takes raw SQL and is **never reused**. Both files are deleted as part of this work; their only live value (`write_dataframe`'s per-backend dispatch) is reimplemented inside the tenant-aware layer.

### 3.5 Source adapters — "any format"

Ingestion is a contract, not a code path with a CSV branch:

```
SourceAdapter (Protocol)
  kind: str
  async discover() -> DiscoveredStructure   # tables, columns, types, keys — no data
  async read(table: str, limit: int | None) -> DataFrame
  supports_incremental: bool
```

Shipping at v1:

| Adapter | Formats / systems | `discover()` yields |
|---|---|---|
| `FileAdapter` | `.csv`, `.parquet`, `.json`, `.xlsx`, `.xls` — everything `ReaderFactory` already registers for polars and pandas | one table; types inferred (Parquet carries real types) |
| `PostgresAdapter` | live customer Postgres | full structure read from `information_schema` — **real** types, PKs, FKs |
| `MySQLAdapter` | live customer MySQL | same |

Adding a format means registering an adapter and nothing else changes. Named as fitting the contract without being built now: JSONL, Avro, ORC, Google Sheets, S3 prefixes, REST endpoints.

`ReaderFactory` already supports the file formats above; the defect is that `POST /v1/sources` and the ingest path assume CSV. That is a fix, not a subsystem.

### 3.6 Multi-source layout and cross-source keys

An org's sources — files, and one or more customer databases — **all land in the same tenant database** (D11). That is what makes a foreign key between two of the customer's own databases enforceable: Postgres enforces FKs across schemas within a database, and never across databases.

The agent decides the layout and records it in the `SchemaSpec`:

- **Merged (default).** Every source becomes a table in `tenant_<hex>`. Table names come from the source's own table name, prefixed with a source alias **only on collision** — two connected databases that both have `users` produce `users` and `crm__users`, not two prefixed names.
- **Namespaced.** When a customer database is large or its structure should stay recognisable, its tables go to `tenant_<hex>_<alias>`. FKs still cross freely, because it is still one database.

The choice is part of the proposal a human accepts, and the rationale field must say why. A merge is not reversible by simply un-merging — renaming tables breaks anything already referencing them — so a change of layout after the fact is a migration proposal like any other.

### 3.7 The read tier

D3 makes SQL the source of truth and Parquet a compute cache; D12 adds a second read path for large tables. Both produce a DataFrame, so nothing downstream knows the difference:

| Table size | Read path | Why |
|---|---|---|
| below threshold | `pl.read_database()` | one dependency, already the simplest path |
| above threshold | DuckDB `postgres` extension → Arrow → polars | streams rather than materialising row-by-row through the driver; the Arrow handoff is zero-copy |
| very large | Spark | infrastructure already exists as the `spark` optional extra; **deliberately not enabled** in this project |

This reintroduces a tuning threshold, which ADR-0013 spent its whole design removing. The distinction that makes it acceptable: ADR-0013's thresholds were **semantic** — they decided what counts as drift, so a wrong value produced wrong answers. This one is purely a **compute** choice with identical results on either side of it, so a wrong value costs latency and nothing else. It is measured, not calibrated, and it belongs in settings rather than in a per-column model.

### 3.8 Component layout

Each unit has one purpose, a stated dependency, and can be tested alone.

| File | Purpose | Depends on |
|---|---|---|
| `engine/src/lumen/architect/spec.py` | `SchemaSpec` and friends; validation | nothing (pure) |
| `engine/src/lumen/architect/infer.py` | type inference, PK selection, FK detection | `materialize.py` (pure) |
| `engine/src/lumen/architect/ddl.py` | `render_ddl`, identifier sanitisation | `spec.py` (pure) |
| `engine/src/lumen/architect/adapters/` | `SourceAdapter` protocol + File/Postgres/MySQL | `ReaderFactory` |
| `engine/src/lumen/datasets/sql_read.py` | the two-path reader (§3.7): polars below threshold, DuckDB above, both returning a DataFrame | polars, duckdb |
| `services/api/src/lumen_api/tenant_db.py` | tenant engine/pool, schema and role provisioning, `tenant_session()` | `db/session.py` |
| `services/api/src/lumen_api/architect.py` | orchestration: design → Proposal; apply → DDL | engine architect, `tenant_db` |
| `services/api/src/lumen_api/credentials.py` | encrypt/decrypt customer DSNs | settings |
| `services/worker/src/lumen_worker/ingest.py` | staging load job, design job | all of the above |

The deterministic core sits in the engine and is DB-free; everything needing a session sits in the API layer. This mirrors how ADR-0011 split `structural_shape()` (pure, unit-tested) from `record_decision()` (needs a session), and ADR-0013 split `baseline.py` from `baselines.py`.

---

## 4. Data flow

```
CONNECT A SOURCE  (file of any supported format, or a customer database)
   │
   ├─ file ──────→ [job] adapter.read() → write to tenant_<hex>_raw
   │                    data is browsable IMMEDIATELY
   │
   └─ database ──→ [job] adapter.discover() only
                        structure mirrored in seconds, real FKs, zero bytes copied
                        customer marks which tables to analyse
                            └─→ [job] adapter.read(table) → tenant_<hex>_raw
   │
   ▼
[job] DESIGN  (deterministic)
   types · uniqueness · containment · naming
   + canonical_entities (ADR-0009) as semantic FK evidence
   + LLM only for readable names, PK tie-breaks, and rationale prose
   │
   ▼
Proposal  kind='schema_design'   ── with ImpactReport (ADR-0010)
   │
   human accepts
   ▼
render_ddl(spec) → ONE transaction → tenant_<hex>
   + artifact_dependencies rows (ADR-0010)
   + data_sources.table_name finally written

RE-INGEST  (Sentinel tick, re-upload, or DB sync)
   schema unchanged  → full replace, one transaction, SET CONSTRAINTS ALL DEFERRED
   schema changed    → additive / type-widening → auto-apply IF trusted (ADR-0011)
                     → narrowing / lossy        → Proposal, always, no exception
                     → column absent at origin  → mark deprecated, never DROP
```

**Staging lifecycle.** `tenant_<hex>_raw` is not a temporary buffer that disappears on promotion — it is the permanent landing zone, and every re-ingest lands there first. Promotion copies from raw into the modelled schema; raw keeps the most recent load of each source and nothing older. Two reasons it is not dropped: the raw-data browser (Project B) reads it for sources whose schema is still awaiting review, and a failed promotion must be retryable without re-downloading the origin. Storage cost is one extra copy of the latest snapshot, which is bounded and predictable.

**Structure-only sources.** When a customer database is connected but no table has been selected yet, the mirrored structure lives **only in the `SchemaSpec`** — no empty tables are created in `tenant_<hex>`. A table appears in the tenant schema exactly when its data is first promoted. Creating empty placeholders was rejected because it makes `information_schema` claim tables that hold nothing, which defeats the point of reading structure from the database rather than from a side table.

**Where the diagram reads from.** These two rules make the diagram a union of two sources, and the distinction is deliberate, not incidental:

| Table state | Read from | Rendered as |
|---|---|---|
| Materialised in `tenant_<hex>` | `information_schema` — enforced types, PKs, FKs | solid, authoritative |
| Known from `discover()`, not yet copied | the accepted `SchemaSpec` | outlined, "available, not imported" |

`information_schema` is authoritative for everything that exists — which is what D1 buys, and why an enforced FK can never be drawn as a relationship the database does not actually hold. The spec supplies only what is known but deliberately not materialised, and the UI must distinguish the two rather than blending them.

---

## 5. Integration with existing work

**`trust.py` must learn new signatures.** `structural_shape()` currently classifies only data-shaped kinds. Without new cases — `schema_design:*`, `schema_migration:additive|type_widening|destructive` — ADR-0011 cannot accrue trust for this pattern and D7's auto-apply never becomes possible. This is a required integration point, not an optional one.

**`HandleStore.resolve()` changes substrate.** It stops downloading Parquet and starts reading the tenant schema through `sql_read.py`'s two-path reader (§3.7). Because the engine's contract is a DataFrame, ADR-0008 through ADR-0013 need no changes — and because DuckDB hands off Arrow, neither path changes what those ADRs receive. Parquet remains as a compute cache and an implementation detail.

**ADR-0009 gains a structural purpose.** `canonical_entities` — "these columns across sources are the same business concept" — becomes one of three FK evidence sources, alongside structural containment and naming. Its human-reviewed status makes it the strongest inferred evidence.

**ADR-0010 applies unchanged.** Accepting a schema change affects downstream artifacts; `artifact_dependencies` already models exactly this, and `compute_impact_report` should run at proposal creation as it does for every other kind.

**A latent bug gets fixed.** `data_sources.row_count` is SELECTed by three read paths and written by none — permanently NULL, which is why the UI renders "— rows". Ingestion now has a real count and writes it.

---

## 6. Security

**Customer database credentials.** `dsn_encrypted` exists with the comment *"Never returned by any read endpoint. Encrypted before it is written"* — and **no encryption machinery exists anywhere in the repo**. This work builds it:

- Symmetric authenticated encryption (Fernet) with a key from settings, never committed, distinct from the Supabase keys.
- Encrypt on write, decrypt only inside the adapter at connection time; the plaintext DSN never enters a response, a log, or an agent's context.
- The connection is opened by the worker, not the API, and never by an LLM-invoked tool.
- Key rotation is out of scope for v1 but the storage format carries a key-id prefix so rotation is possible later without a migration.

**Blast radius.** Two layers, and they fail independently. The separate instance (§3.1) means the control plane is not reachable from tenant code at all — there is no connection to it. The per-org role (§3.2) means one tenant cannot reach another *within* the tenant instance. Both get dedicated integration tests (§8); the second is the one that would silently rot, because a bug in it produces no symptom until it produces a breach.

**Row exposure.** Customer row data crosses the HTTP boundary for the first time in this product. That surface belongs to Project B (the raw-data browser) and is explicitly not designed here; this project only makes the rows exist and queryable server-side.

---

## 7. Error handling

| Failure | Behaviour |
|---|---|
| Malformed / unreadable file | `data_sources.status='error'`, no proposal, message surfaced on the source |
| Unsupported format | Rejected at connect time with the list of supported formats — never a partial ingest |
| Customer DB unreachable | Source marked `error`; structure retained from last successful discover; retried on the next tick |
| Bad DB credentials | `error` with an explicit "credentials rejected" state, distinct from unreachable |
| DDL fails | Transaction rolls back; proposal → `failed`; tenant schema unchanged |
| Constraint violated on re-ingest | `DriftEvent` (D5); transaction rolls back; table keeps its previous contents |
| Type inference ambiguous | Falls back to `text` and records the ambiguity in the spec's rationale |
| Quota exhausted | No proposal is created; **staged data stays usable** — the customer is not blocked from seeing their data by a billing state |
| No API key configured | The deterministic path produces a valid `SchemaSpec` anyway; only names and prose degrade |

---

## 8. Testing

**Unit (no database, default suite):** dtype → `SqlType` mapping; PK selection including ties and the no-viable-PK case; FK detection over synthetic frames (total containment, partial containment, no relationship, self-reference); identifier sanitisation (collisions, reserved words, leading digits, over-63-byte names, unicode); `render_ddl` against golden strings; `SchemaSpec` validation rejecting malformed specs; the reversibility classifier for migrations.

**Integration (live, against *both* instances — Supabase for the control plane and the tenant Postgres for customer data). The existing `-m integration` suites only ever needed one connection; the fixtures gain a second, and a run with the tenant DSN unset must skip rather than fail confusingly:**

1. **Tenant isolation — the critical test.** Under `tenant_A_role`, `SELECT` against `tenant_B.*` must fail. And the tenant instance must contain no control-plane table at all — assert that `organizations`, `api_keys`, and `subscriptions` do not exist there, so the separation is verified as a fact about the instance rather than trusted as a configuration. If these tests do not exist, the design is not implemented.
2. Provisioning is idempotent — bootstrapping twice does not error or duplicate.
3. Staging load for each supported file format.
4. `discover()` against a real Postgres source reproduces its declared PKs and FKs exactly.
5. Accept a `schema_design` proposal → DDL executes → tables, types, PK, and enforced FKs all exist as specified.
6. Re-ingest with unchanged schema replaces contents inside one transaction without violating FKs.
7. Additive migration auto-applies when trust is earned, and awaits review when it is not (mirroring `test_sentinel_diagnosis.py`'s existing pair).
8. A destructive migration is always a proposal, even with maximum trust.
9. A constraint violation on re-ingest produces a `DriftEvent` and leaves the table intact.
10. Credentials round-trip through encryption and never appear in any API response.
11. **Cross-source FK enforcement (D11).** Two *separate* customer databases are connected; an FK is declared from a table originating in one to a table originating in the other; it is enforced, and an insert violating it is rejected by Postgres. This is the test that proves the "FK crosses between two of the customer's databases" requirement is real rather than drawn.
12. **Read-path parity (D12).** The same table read below and above the threshold produces frames with identical schema, row count, and contents. A tiering that changes results is a correctness bug, not a performance knob.
13. **Layout choice round-trips.** A merged layout and a namespaced layout each produce a working schema, and a name collision between two sources resolves to the documented `alias__table` form rather than overwriting.

---

## 9. Out of scope

Deliberately excluded from this project, with the reason:

- **Normalisation / splitting tables** — D2 chose conservative modelling; revisit once trust and evidence exist.
- **A SQL console for the customer** — belongs to Project B/C.
- **The raw-data browser and the ERD screen** — Project B. This project makes them *possible*; it does not build them.
- **Temporal history (SCD)** — D6; ADR-0013's baselines already answer "what changed".
- **Incremental sync for live databases** — v1 refreshes selected tables in full. A watermark-based incremental path is the obvious follow-up and the adapter contract already carries `supports_incremental` for it.
- **Backfilling sources that exist only as Parquet today** — there is one demo source; a one-off backfill is cheaper than a migration path.
- **Credential key rotation** — storage format allows it; the tooling is not built.

---

## 10. Deferred questions

- **Storage cost attribution.** Tenant schemas consume storage on the new instance that no plan limit currently meters. `compute_seconds` exists as an ungated metric with no plan-limit column; a `storage_bytes` metric belongs to the same gap and to ADR-0004's territory, not this one. This gets sharper with a dedicated instance, because the cost is now a line item rather than part of the Supabase bill.
- **Where the read threshold sits.** §3.7 states the tiering; the actual row count at which DuckDB beats `pl.read_database()` is unmeasured and must be benchmarked rather than guessed. Until it is, a conservative default with a settings override is correct.
- **Provisioning the tenant instance.** This design assumes the instance exists and its DSN is configured. Who creates it, how it is backed up, and how it is migrated when the schema-provisioning logic itself changes are operational questions this spec does not answer and the plan must not silently invent.
- **Connection limits.** One pool against the tenant instance is assumed. Postgres default `max_connections` is 100; per-org roles do not multiply connections, but concurrent worker jobs do, and `max_jobs=10` in arq is the current ceiling. Worth measuring before it becomes an incident.
