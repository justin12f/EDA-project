# ADR-0006: Supabase as the platform, pgvector as the agent's memory

**Status:** Accepted
**Date:** 2026-08-04
**Supersedes:** [ADR-0002](ADR-0002-multi-tenancy-and-auth.md) (identity and session mechanics only — its tenancy model and RLS-in-the-database principle survive intact)
**Deciders:** Project owner

## Context

ADR-0002 chose to build identity in-house: Argon2id hashing, opaque sessions in Redis, our own signup/login endpoints, and a self-hosted Postgres reached by a bespoke `lumen_app` role. That decision was made when the alternative was evaluated as "Supabase covers only half the system, because the Python worker still needs direct Postgres access."

The owner has since directed that Supabase be used for the database, authentication and sign-in. A second requirement arrived with it: **the data contexts the agent produces must be stored in a vector database** so they can be retrieved semantically rather than by exact key.

That second requirement changes the arithmetic behind the original decision. The system now needs a Postgres with `pgvector`, an embedding pipeline, and an auth system — and Supabase ships all three as one managed product with RLS as its native tenancy mechanism. The "only half the system" objection was about auth alone; it does not hold once the vector store is in scope, because the worker's direct-Postgres path is exactly how you talk to pgvector anyway.

Forces:

- Every byte of identity we do not write is identity we do not have to get right. Password reset, email verification, OAuth, MFA and session revocation are each a small project.
- The Python worker must reach Postgres directly — for Spark/Polars writes, for pipeline execution, and for vector search. It cannot go through PostgREST.
- Tenant isolation must still be enforced *in the database*. That requirement from ADR-0002 is not relaxed; Supabase RLS is how it is met.
- The product must keep running with no LLM API key (see [ADR-0003](ADR-0003-agent-orchestration.md) and the keyless providers). Embeddings must therefore not require a paid key either.

## Decision

### 1. Supabase is the database and the identity provider

| Concern | Owner |
|---------|-------|
| Postgres 15 + `pgvector` + `pg_trgm` | Supabase |
| User accounts, email/password, OAuth, email verification, password reset, MFA | Supabase Auth (GoTrue) |
| JWT issuance and refresh | Supabase Auth |
| Organizations, memberships, roles | **our** tables in `public`, under RLS |
| Data sources, runs, proposals, agent events, usage, dataset handles, data contexts | **our** tables in `public`, under RLS |
| File storage for uploads and Parquet artifacts | Supabase Storage |

The `AuthProvider` seam that ADR-0002 introduced is what makes this a swap rather than a rewrite: `SupabaseAuthProvider` replaces `LocalAuthProvider` behind the same interface.

### 2. RLS is keyed on the Supabase JWT, not on a bespoke session variable

ADR-0002 set `app.current_org` per transaction. That is replaced by the Supabase-native pattern, because it is what `auth.uid()` and every Supabase policy example already assume:

```sql
-- opened by the API for every request that carries a user
SET LOCAL ROLE authenticated;
SET LOCAL request.jwt.claims = '{"sub":"<user-uuid>","role":"authenticated"}';
```

Policies then read `auth.uid()`. Membership is resolved through one `SECURITY DEFINER` helper so policies stay one-liners and cannot recurse:

```sql
CREATE FUNCTION public.is_org_member(target uuid) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.memberships m
    WHERE m.org_id = target AND m.user_id = auth.uid()
  );
$$;

CREATE POLICY runs_org_isolation ON public.runs
  USING (public.is_org_member(org_id))
  WITH CHECK (public.is_org_member(org_id));
```

The API and the worker connect as `postgres` (service role) but **always** open tenant work through a helper that sets `ROLE authenticated` and the JWT claims for that transaction. The service role is used unimpersonated only for signup-time bootstrap and for background jobs that legitimately span tenants — and those call sites are enumerated in one module, not scattered.

### 3. The agent's context lives in pgvector

Every artefact the agent learns about a dataset — a profile, a schema summary, a cleaning rationale, a decision a human made — is written to `data_contexts` with an embedding:

```
data_contexts
  id, org_id, source_id, rid
  kind       profile | schema | rationale | decision | note
  content    the text that was embedded
  embedding  vector(384)
  metadata   jsonb
  created_at
```

with an HNSW index on `embedding vector_cosine_ops`. A `match_data_contexts(query_embedding, match_org, match_count, min_similarity)` RPC does the search, and the agents get a `search_data_context` tool over it.

This is what turns a stateless agent into one with memory: on the second visit to a source it retrieves what it concluded the first time, including which proposals the human rejected and why.

### 4. Embeddings are computed locally

Neither Anthropic nor Groq — the two configured LLM providers — offers an embedding endpoint. Rather than add a third vendor and a third key, embeddings come from **`fastembed`** running `BAAI/bge-small-en-v1.5` (384 dimensions, ONNX, CPU, ~130MB, no network at inference time).

This keeps the no-API-key guarantee whole: a fresh checkout profiles data, embeds it, stores it, and retrieves it with zero credentials. The `EmbeddingProvider` interface leaves room for a hosted model later; the vector dimension is the only thing a swap would force, and it is a settings value plus one migration.

## Options Considered

### Option A — Keep ADR-0002's in-house auth, add pgvector to self-hosted Postgres

| Dimension | Assessment |
|-----------|------------|
| Complexity | High — auth is ours forever |
| Cost | Server cost only |
| Time to production | Slowest |
| Control | Total |

**Pros:** no vendor in the critical path; the tenancy design already written stays untouched; one less external failure domain.
**Cons:** password reset, email verification, OAuth, MFA and session revocation are all still unwritten; a solo maintainer running auth in production is carrying real security risk for no product differentiation. Contradicts the owner's direction.

### Option B — Supabase for database + auth + storage, pgvector for context (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost | Free tier, then usage-based |
| Time to production | Fastest credible |
| Control | Good — it is plain Postgres underneath, and the schema is ours |

**Pros:** auth arrives complete, including the flows nobody remembers to build; `pgvector` and Storage are in the same product, so there is one connection string rather than four; RLS is Supabase's native idiom, so the isolation requirement from ADR-0002 is *better* served, not worse; the Python worker still gets a plain `postgresql://` DSN, so nothing about the engine changes.
**Cons:** a vendor in the critical path; local development needs the Supabase CLI (or a hosted dev project); `auth.users` lives in a schema we do not own, so every join to it goes through our `profiles` mirror; migrating off later means reimplementing auth — the thing Option A would have already had.

### Option C — Supabase for auth only, separate Postgres for data and vectors

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost | Two systems |
| Control | Good |

**Pros:** keeps heavy analytical writes off the auth database; failure domains stay separate.
**Cons:** `auth.uid()` does not exist in the other database, so RLS has to be re-invented there — reintroducing exactly the bespoke session-variable machinery this ADR is replacing. Two connection strings, two migration histories, and cross-database joins that cannot happen. The isolation story gets worse in the database that actually holds the tenant data.

### Option D — A dedicated vector database (Pinecone, Qdrant, Weaviate)

**Pros:** better recall tuning at scale; purpose-built index management.
**Cons:** a third datastore and a second consistency problem — a context row and its embedding could no longer be written in one transaction, which is the same "recorded but unbilled" failure mode ADR-0004 rejected for usage. At the volume of one profile per dataset per run, `pgvector` with HNSW is not the bottleneck. Revisit past ~10⁶ vectors.

## Trade-off Analysis

The decisive fact is that **the vector requirement and the auth requirement want the same database**. A data context is scoped to an org and a source; if it lives in a different system from `data_sources`, then either its isolation is enforced somewhere new (Option C's problem) or it is not enforced at all (Option D's). Keeping vectors in the same Postgres as the rows they describe means one RLS policy covers both, and a context row is written in the same transaction as the run that produced it.

Against Option A, the honest reckoning is that in-house auth was the right call under ADR-0002's assumptions and is the wrong call under these. The objection recorded there — "Supabase covers only half the system" — was correct about auth alone and stops being correct once Postgres, Storage and pgvector are all in scope. What survives from ADR-0002 is the part that mattered: isolation is enforced by the database, verified by a test that enumerates `pg_tables` and fails on any table without `rowsecurity`. That test is carried over unchanged.

The real cost accepted here is vendor concentration: auth, data, vectors and blobs now share one provider and one outage. That is mitigated by the schema being ours and portable — every table, policy and function lives in versioned SQL in this repository, and the worker speaks plain `postgresql://`. What would not be portable is auth, and that is the deliberate trade.

Local embeddings are the one place this ADR spends complexity to buy independence. A hosted embedding API would be three lines; `fastembed` is a 130MB model download and a warm-up cost on first run. It is worth it because it preserves the property that makes this codebase pleasant to work on: `git clone`, `make dev`, and the whole product runs — no account, no key, no card.

## Consequences

**Easier**
- Sign-in, sign-up, password reset, email verification and OAuth: configuration, not code.
- Semantic recall across runs — "what did we conclude about this source last week" becomes a query.
- One connection string for the API, the worker, and the vector search.
- Storage for uploads and Parquet artefacts with per-org path prefixes and the same identity model.

**Harder**
- Local development needs the Supabase CLI (`supabase start`) or a hosted dev project. The old `docker compose up postgres` is no longer the whole story.
- `auth.users` is not ours. A `profiles` table mirrors the subset we need (`display_name`, `avatar_url`) and is kept in sync by a trigger on `auth.users`.
- Two places can now create a user (Supabase Auth and our bootstrap), so org creation on first sign-in must be idempotent.
- The first run downloads an embedding model. CI needs that cached or the fixture set stubbed.

**To revisit**
- A dedicated vector store past ~10⁶ vectors per org, or if recall tuning becomes a product concern.
- A hosted embedding model if `fastembed`'s CPU cost shows up in worker latency.
- Self-hosting Supabase if vendor concentration becomes unacceptable — the schema is portable by construction; only GoTrue configuration would move.

## Action Items

1. [ ] Add `supabase/` with `config.toml` and versioned SQL migrations; drop the bespoke `lumen_app`/`lumen_migrator` roles from `infra/postgres/init/`.
2. [ ] Write `0001_identity.sql`: `profiles` (mirroring `auth.users` via trigger), `organizations`, `memberships`, `is_org_member()`, and the org bootstrap function.
3. [ ] Write `0002_domain.sql`: `data_sources`, `runs`, `proposals`, `agent_events`, `usage_records`, `dataset_handles` — every one with `org_id` and an `is_org_member` policy.
4. [ ] Write `0003_vector.sql`: `CREATE EXTENSION vector`, `data_contexts`, the HNSW index, and `match_data_contexts()`.
5. [ ] Replace the API's session dependency with Supabase JWT verification; keep `current_org_id` and `require_role` signatures unchanged so downstream routers do not move.
6. [ ] Add `engine/src/lumen/embeddings/` with `EmbeddingProvider` and a `fastembed` implementation.
7. [ ] Add the `search_data_context` and `write_data_context` tools to the agent tool registry.
8. [ ] Carry over the `test_rls.py` table sweep unchanged, retargeted at the Supabase database.
9. [ ] Frontend: `@supabase/supabase-js` + `@supabase/ssr` for sign-in, with the JWT forwarded to the API.
