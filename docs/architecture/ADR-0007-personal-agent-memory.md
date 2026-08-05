# ADR-0007: What the agent remembers, and who it remembers it for

**Status:** Accepted
**Date:** 2026-08-04
**Extends:** [ADR-0006](ADR-0006-supabase-and-vector-context.md)
**Deciders:** Project owner

## Context

ADR-0006 introduced `data_contexts`: everything the agent learns about a dataset,
stored with an embedding so it can be retrieved semantically. It scoped every row
to the organization.

The direction since is that the memory must be **personalised per individual
user**. Taken literally — scope everything to `user_id` — that is a worse product
than what it replaces, and the reason is worth stating before the design.

A workspace has two kinds of remembered thing, and they have opposite sharing
requirements:

**A profile is a measurement.** "`country_code` is 3.2% null across 12.4M rows"
is a property of the data, not an opinion. Scoping it per user means the second
teammate to open that source pays the full scan again to learn the identical
number — and the third, and the fourth. Worse, they may get a *different* number
if the data moved between scans, and now two people hold contradictory beliefs
about a shared table with no way to notice.

**A rejection is a judgment.** "Never deduplicate orders by `customer_id` —
repeat orders are legitimate" is true about the person's understanding of the
business. Sharing it org-wide means one analyst's call silently steers everyone
else's agent. The next person gets recommendations shaped by reasoning they never
saw, cannot question, and may disagree with. That is not personalisation; it is
one user's preferences leaking into everyone's tooling.

So the useful question is not *whose row is this* but **what kind of claim is
it** — a fact about the data, or a fact about a person.

## Decision

`data_contexts` gains `user_id` and a `scope` of `org` or `user`.

| Scope | Holds | Visible to | Why |
|-------|-------|------------|-----|
| `org` | profiles, schemas, applied-pipeline rationales | every member of the organization | measured once, identical for everyone, expensive to recompute |
| `user` | decisions, rejections, preferences, notes | **only their author**, even inside the same organization | a colleague's judgment is not evidence about your intent |

The scope is **derived from the kind**, not chosen per call site:

```python
DEFAULT_SCOPE = {
    Kind.PROFILE:   Scope.ORG,
    Kind.SCHEMA:    Scope.ORG,
    Kind.RATIONALE: Scope.ORG,
    Kind.DECISION:  Scope.USER,
    Kind.NOTE:      Scope.USER,
}
```

"Is this about the data or about the person" is exactly the judgment that gets
made inconsistently when it is made at twenty call sites. Deciding it once, in a
table a reviewer can read in ten seconds, is the difference between a rule and a
convention. A caller may still override it explicitly, which keeps the unusual
case possible without making it the default.

### Isolation lives in the policy, not in the code

```sql
create policy data_contexts_read on public.data_contexts
  for select using (
    public.is_org_member(org_id)
    and (scope = 'org' or user_id = auth.uid())
  );
```

The Python store cannot leak one person's history into another's context even if
it tries, because every read runs inside that user's RLS session. This matters
more here than in most tables: the consumer of these rows is a language model
that will happily use whatever it is handed. A forgotten `WHERE` clause in a
retrieval path would not raise, would not fail a test that checks output shape,
and would surface as an agent that occasionally references a decision the user
never made.

Two further constraints, both about failure modes rather than features:

- `check (scope = 'org' or user_id is not null)` — a user-scoped row with no
  author would be invisible to everyone including its creator. That is silent
  data loss wearing the costume of a safe default.
- The write policy requires `user_id = auth.uid()`, so one member cannot author
  private context as another. Tested, not assumed.

### Retrieval prefers your own memory

`match_data_contexts` takes `prefer_personal` (default `0.05`) added to the
similarity of rows you authored. It biases **ranking**, never visibility — the
policy has already decided what you can see.

The effect is what makes a second visit feel like the agent remembers *you* and
not merely the table: when your note and a shared profile are equally relevant,
your note comes first. The value is deliberately small. A large bias would let
stale personal notes bury a fresh measurement, which is the failure mode of every
recommender that over-weights history.

### The briefing

`personal_source_context(org, source)` returns the most recent shared fact of
each kind plus that person's own history, and the store renders it as markdown
the agent reads before its first tool call:

```markdown
## What is already known about this source
- **Profile of orders.csv** — 1.8M rows, 5 columns. Null rates above 0.5%: currency 4.1%.

## What you previously decided here
- 2026-08-04 · **No dedupe** — Repeat orders are legitimate.
```

Exact recall by key, not similarity search. When you know which source you are
looking at, an approximate nearest-neighbour query is a worse way to answer the
question than a lookup.

## Options Considered

### Option A — Keep everything org-scoped (the status quo)

| Dimension | Assessment |
|-----------|------------|
| Complexity | None — already built |
| Personalisation | None |
| Failure mode | One person's judgment steers everyone |

**Pros:** simplest possible model; profiles are shared, which is right; nothing
to migrate.
**Cons:** does not do what was asked. Rejections and preferences become org-wide
policy by accident, and the agent gets less useful to everyone as more people use
it — each new opinion dilutes the last.

### Option B — Scope everything to the user

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Personalisation | Total |
| Failure mode | Every teammate re-derives the same facts |

**Pros:** trivially satisfies "personalised per user"; the strongest possible
privacy story; one predicate to reason about.
**Cons:** turns a shared workspace into N private ones. The second person to
open a 48M-row source pays the full profiling cost to learn a number their
colleague measured an hour ago, and the team has no shared understanding of its
own data. It also makes the product worse the more a team collaborates, which is
backwards for something sold to teams.

### Option C — Scope by the nature of the claim (chosen)

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium — two scopes, one policy, a default table |
| Personalisation | Where it means something |
| Failure mode | A miscategorised kind, caught by the default table |

**Pros:** facts are measured once and shared; judgments stay with their author;
the distinction is enforced in the database and centralised in one dict rather
than scattered across call sites.
**Cons:** somebody must decide the scope for each new kind, and getting that
wrong is a real (if small and reviewable) mistake. It is also more than was
literally asked for, which is a cost if the reasoning above turns out not to
match how these teams actually work.

### Option D — Per-user overlay on shared context

Personal rows *override* org rows of the same kind rather than sitting alongside
them.

**Pros:** a single resolved view; no ranking heuristic needed.
**Cons:** an override silently hides the shared fact, so a user with a stale
personal note stops seeing the current measurement and has no signal that it
changed. The bug is invisible precisely when it matters. Rejected for that
alone.

## Trade-off Analysis

The decisive question is what happens as a team grows. Option B optimises the
first user and penalises every subsequent one; Option A optimises the group and
penalises the individual. Option C is more machinery than either, and the
machinery buys the property that both of those lack: **adding a teammate makes
the shared knowledge better and leaves each person's judgment their own.**

The honest cost is a new judgment call per context kind. That is mitigated by
centralising it — `DEFAULT_SCOPE` is five lines, a test asserts every kind has a
decided default, and getting one wrong is visible in review rather than buried in
whichever handler wrote the row.

The `prefer_personal` bias is the one piece here chosen by taste rather than
argument. `0.05` is small enough that a much better-matching shared fact still
wins and large enough to break ties toward your own history. It is a settings
value, and if it turns out to be wrong it is wrong in a way that shows up as
"the agent keeps bringing up something old" rather than as a correctness bug.

## Consequences

**Easier**
- A second visit to a source starts from what was already measured, and from what
  *you* decided — including proposals you rejected, so the agent stops re-proposing
  them.
- Onboarding a teammate: they inherit the workspace's understanding of its data
  without inheriting anyone's opinions.
- Answering "why did the agent suggest this": the retrieved context is rows with
  authors and timestamps, not an opaque prompt.

**Harder**
- Every new context kind needs a scope decision. Enforced by a test, not by memory.
- Retrieval is now two-dimensional (similarity × scope), so a recall complaint has
  two possible causes.
- Deleting a user cascades their private context away. That is correct, and it
  means a departing analyst's reasoning leaves with them — which teams should know
  before they rely on it.

**To revisit**
- A `team` scope between the two, if organizations grow large enough that a
  sub-team's conventions should be shared but not global.
- Letting a user *promote* one of their decisions to org scope explicitly. The
  data model already supports it; what is missing is the UI and the question of
  who may approve it.
- Whether `prefer_personal` should decay with age, so a two-year-old note stops
  outranking a fresh measurement.

## Action Items

1. [x] Migration `20260804000004_personal_context.sql` — `user_id`, `scope`, the
       author check, four policies, and `prefer_personal` in `match_data_contexts`.
2. [x] `personal_source_context()` returning shared facts plus personal history.
3. [x] `ContextStore` with `remember`, `remember_many`, `search`, `briefing`.
4. [x] Agent tools `recall_context` and `remember_decision`.
5. [x] `profile_source` writes an org-scoped profile automatically.
6. [x] Integration tests, including two teammates in one organization.
7. [ ] Wire `briefing()` into the agent prompt as the first thing it reads.
8. [ ] Backfill embeddings for rows written while the embedder was unavailable.
9. [ ] Surface "the agent remembered this" in the UI, with the source row visible.
