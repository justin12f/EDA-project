-- ADR-0012: collective immune memory. `global_patterns` deliberately has no
-- `org_id` column at all — not a policy promising isolation, a schema that
-- cannot represent the thing it must never leak. There is no query, correct
-- or buggy, that can reconnect a shared pattern to the organization that
-- produced it, because the column to do that with does not exist.
--
-- `org_pattern_contributions` is the opposite: an ordinary org-scoped
-- ledger, isolated like every other tenant table in this schema. Its only
-- job is the mechanism behind Decision §4 ("aggregate, don't trust a single
-- contributor") — capping what any one org can add to the anonymous
-- aggregate at exactly one occurrence and one outcome per (org, pattern,
-- fix) signature pair, ever, regardless of how many times its own Sentinel
-- re-encounters that shape. See `lumen_api/global_patterns.py`'s `_claim()`
-- for the upsert that turns this table into that limit.

alter table public.organizations
  add column if not exists pattern_contribution_enabled boolean not null default false;

create table if not exists public.global_patterns (
  id                uuid primary key default gen_random_uuid(),
  pattern_signature text not null,
  fix_signature     text not null,
  occurrences       integer not null default 0,
  applied_count     integer not null default 0,
  rejected_count    integer not null default 0,
  -- Redundant with applied_count/rejected_count by design (same reason
  -- pattern_trust_scores stores both the raw counts and a derived `score`):
  -- recomputed on every write in application code, so a read never has to
  -- redo the division or special-case a zero denominator.
  success_rate      real not null default 0,
  first_seen_at     timestamptz not null default now(),
  last_seen_at      timestamptz not null default now(),
  unique (pattern_signature, fix_signature)
);

-- Enabled and forced, with zero policies — not "isolated per org" (there is
-- no org_id to isolate by) but "reachable by nobody at all except the
-- service-role connection `service_session()` opens", the same way every
-- other service_session-read table in this schema (data_source_schedules,
-- drift_events, ...) is already forced without that blocking service_session
-- itself. Every read and write to this table in application code goes
-- through lumen_api/global_patterns.py; no client request reaches it
-- directly, and no user_session ever could even if one tried.
alter table public.global_patterns enable row level security;
alter table public.global_patterns force row level security;

create table if not exists public.org_pattern_contributions (
  org_id                 uuid not null references public.organizations (id) on delete cascade,
  pattern_signature      text not null,
  fix_signature          text not null,
  contributed_occurrence boolean not null default false,
  contributed_outcome    boolean not null default false,
  created_at             timestamptz not null default now(),
  primary key (org_id, pattern_signature, fix_signature)
);

create index if not exists org_pattern_contributions_org_idx on public.org_pattern_contributions (org_id);

alter table public.org_pattern_contributions enable row level security;
alter table public.org_pattern_contributions force row level security;

do $$ begin
  create policy org_pattern_contributions_org_isolation on public.org_pattern_contributions
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;
