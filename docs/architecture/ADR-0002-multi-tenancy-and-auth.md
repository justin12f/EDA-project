# ADR-0002: Multi-tenancy, identity and data isolation

**Status:** Proposed
**Date:** 2026-08-03
**Deciders:** Project owner

## Context

There is no notion of a user anywhere in the codebase. `PostgresManager` connects as a single `admin` superuser to a single `eda_db` and writes with `if_exists="replace"` — any caller can overwrite any table. `api/supabase_api.py` is an empty file, suggesting Supabase was considered.

The product handles two very different classes of data:

1. **Control-plane data** — accounts, organizations, memberships, data-source definitions, runs, proposals, usage records. Small, relational, ours.
2. **Tenant payload data** — the customer's actual rows. Potentially tens of millions of rows per source, arriving as uploads (`users_2024.csv`) or as live connections the customer owns (`Production Postgres`, `Analytics MySQL` in the UI).

These have opposite requirements. Control-plane data wants strong relational guarantees in one place. Payload data wants isolation, size headroom, and the ability to be deleted wholesale when a customer leaves.

A leak between tenants is the failure that ends the product. Application-level `WHERE org_id = ?` is one forgotten clause away from that failure.

## Decision

**Tenancy model:** organization-scoped. `User` ↔ `Membership` ↔ `Organization`. Every resource hangs off `org_id`. Personal accounts are an organization of one. Roles: `owner`, `admin`, `member`, `viewer`.

**Control plane:** one Postgres database, shared schema, `org_id` on every table, **PostgreSQL Row-Level Security enforced**. The application connects as a non-superuser role with `BYPASSRLS` withheld, and sets `SET LOCAL app.current_org = :org_id` at the start of every transaction. RLS policies compare against `current_setting('app.current_org')`.

**Tenant payload:** one **schema per organization** (`tenant_<org_id>`) inside the same Postgres for materialized outputs, plus object storage prefixed `s3://<bucket>/org/<org_id>/…` for uploads and artifacts. Customer-owned external sources (their Postgres, their MySQL) are never copied wholesale — only sampled and profiled, with credentials stored encrypted.

**Identity:** email + password with Argon2id, plus OAuth (Google, GitHub) — implemented on our own `services/api` behind a thin `AuthProvider` interface, so a hosted provider can be swapped in later without touching call sites. Sessions are opaque server-side tokens in Redis, delivered to the browser as `HttpOnly; Secure; SameSite=Lax` cookies by `apps/web`. The browser never receives a service token or a raw API key.

**Credential storage:** external data-source secrets are encrypted at rest with envelope encryption (per-org data key, wrapped by a KMS master key) and are never returned by any read endpoint — only a masked form, exactly as the UI's `postgres://lumen:****@…` string implies.

## Options Considered

### Option A — Application-level filtering only (`WHERE org_id = ?`)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | None |
| Isolation strength | Weak — one missing predicate is a cross-tenant leak |
| Team familiarity | High |

**Pros:** nothing to learn; works with any ORM; trivially portable off Postgres.
**Cons:** the guarantee lives in every query an agent or a future contributor writes. This system *generates SQL with an LLM* — an unfiltered generated query is not a hypothetical.

### Option B — Shared schema + Postgres RLS (chosen for the control plane)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Cost | None beyond care with connection pooling |
| Isolation strength | Strong — the database refuses to return other tenants' rows |
| Team familiarity | Medium |

**Pros:** the guarantee is enforced below the application, so LLM-generated SQL and hand-written SQL are both contained; migrations stay single-copy; cross-org admin queries remain possible through an explicitly privileged role.
**Cons:** requires discipline with pooled connections (`SET LOCAL` inside the transaction, never a bare `SET`); a misconfigured pool that reuses a session leaks context; policies must be written for every new table.

### Option C — Database per tenant

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost | High — connection pools and migrations multiply |
| Isolation strength | Strongest |
| Team familiarity | Low |

**Pros:** deletion is `DROP DATABASE`; noisy-neighbour problems disappear; easiest story for an enterprise security review.
**Cons:** running a migration across N databases is a job, not a command; connection pool exhaustion arrives early; a solo maintainer will eventually have databases at different schema versions.

### Option D — Delegate the whole problem to Supabase

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low to start |
| Cost | Predictable, then steep |
| Isolation strength | Good (it is Postgres RLS underneath) |
| Team familiarity | Medium — `api/supabase_api.py` exists but is empty |

**Pros:** auth, RLS, storage, and realtime in one afternoon; excellent for reaching a first paying customer.
**Cons:** the Python worker still needs direct Postgres access, so the Supabase client is only half the system; pushes the tenant warehouse into a database sized for app workloads; the tightest coupling in the product ends up in a vendor.

## Trade-off Analysis

Option B and Option C differ only in *where* isolation is enforced, and both are sound. The deciding factor is operational load on one maintainer: Option C's per-tenant migrations are a recurring tax paid forever, while Option B's cost is a one-time investment in getting the session-variable pattern right plus a policy per table. That policy can be enforced by a test that fails if any table lacks RLS — a check, not a habit.

Option D is genuinely attractive for speed and is not foreclosed: the `AuthProvider` interface exists precisely so Supabase Auth can be dropped in behind it. What is rejected is *depending on Supabase for tenant isolation*, because the worker bypasses it entirely.

Separating the payload into per-org schemas rather than shared tables is worth the extra DDL: it makes `DROP SCHEMA tenant_x CASCADE` a complete, verifiable deletion — which is what a deletion request actually requires.

## Consequences

**Easier**
- Answering "can tenant A see tenant B's data" with a policy and a test rather than a code review.
- Deleting a customer completely: `DROP SCHEMA`, delete the S3 prefix, delete control-plane rows by `org_id`.
- Containing LLM-generated SQL, which is now bounded by the same policies as everything else.

**Harder**
- `database/postgres_manager.py` must stop connecting as `admin` and stop using `if_exists="replace"` against unqualified table names. Both are rewritten to be org-scoped and schema-qualified.
- Every new table needs a policy and a migration. Enforced by a test that enumerates `pg_tables` and asserts `rowsecurity`.
- Connection pooling needs care: `SET LOCAL` inside an explicit transaction, and a pool that never hands a session to a second org without a reset.

**To revisit**
- Moving the largest tenants to dedicated databases once one customer's payload dominates. The per-schema layout makes that a migration, not a redesign.
- Whether to adopt a hosted auth provider once signup volume justifies the operational relief.

## Action Items

1. [ ] Write migrations for `users`, `organizations`, `memberships`, `sessions`, `data_sources`, `runs`, `proposals`, `usage_records`, `audit_log` — each with `org_id` and an RLS policy.
2. [ ] Create the application DB role without `BYPASSRLS`; keep a separate migration role.
3. [ ] Implement a transaction dependency in `services/api` that issues `SET LOCAL app.current_org`.
4. [ ] Add `test_rls.py`: for every table in the control plane, assert `rowsecurity = true` and assert a cross-org read returns zero rows.
5. [ ] Implement `AuthProvider` with a local email/password implementation; wire OAuth behind the same interface.
6. [ ] Implement envelope encryption for data-source credentials; ensure read endpoints return masked values only.
7. [ ] Rewrite `PostgresManager` to take an org context and write into `tenant_<org_id>` with an explicit write mode.
