-- ADR-0011: progressive trust, learned per (org, pattern) from accept/reject
-- history — a second, independent gate on top of ADR-0008's structural
-- confidence ladder, never a replacement for it.
--
-- Two columns beyond the ADR's own schema sketch (approvals, rejections,
-- score, last_decided_at), both needed to implement behavior the ADR states
-- in prose but doesn't carry into its table definition:
--
-- consecutive_approvals — §4's actual mechanism ("any rejection immediately
--   revokes eligibility... requires a fresh run of approvals before it's
--   reconsidered") is a *streak*, not a lifetime ratio. A lifetime Wilson
--   score computed from `approvals`/`rejections` alone recovers only
--   asymptotically after one rejection — it never really resets, which
--   contradicts "fresh run... reconsidered." So this file keeps both: the
--   lifetime counts drive `score`, an honest descriptive statistic an owner
--   reviews (§5 — "how many decisions each is based on"); this streak,
--   which resets to 0 on any rejection, is what the auto-apply gate itself
--   actually reads.
--
-- auto_apply_enabled — §5's explicit kill switch ("a toggle to disable
--   auto-apply for any pattern regardless of score"). Defaults true: the
--   structural-confidence and streak gates are already conservative: this
--   column exists for an owner to override them, not to duplicate them.

create table if not exists public.pattern_trust_scores (
  id                    uuid primary key default gen_random_uuid(),
  org_id                uuid not null references public.organizations (id) on delete cascade,
  pattern_signature     text not null,
  approvals             integer not null default 0,
  rejections            integer not null default 0,
  consecutive_approvals integer not null default 0,
  score                 real not null default 0,
  auto_apply_enabled    boolean not null default true,
  last_decided_at       timestamptz,
  created_at            timestamptz not null default now(),
  -- One row per pattern per org — the posterior update is an upsert keyed
  -- on exactly this.
  unique (org_id, pattern_signature)
);

create index if not exists pattern_trust_scores_org_idx on public.pattern_trust_scores (org_id);

alter table public.pattern_trust_scores enable row level security;
alter table public.pattern_trust_scores force row level security;

-- Org isolation only — any member's accept/reject writes here as a side
-- effect of `decide_proposal` (itself already gated at "member"); the
-- owner-only *toggle* for auto_apply_enabled is enforced in the API layer
-- (services/api/src/lumen_api/trust.py), the same split proposals.py
-- already uses for kind-specific gates that RLS has no column-level way to
-- express.
do $$ begin
  create policy pattern_trust_scores_org_isolation on public.pattern_trust_scores
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;
