-- ADR-0008: scheduled drift detection, diagnosis, and self-healing pipelines.
--
-- One schedule per source (unique constraint below) — the ADR's own examples
-- and its "a schedule toggle per source" frontend item both talk about a
-- single schedule, never a list to manage. `proposals.kind = 'pipeline_patch'`
-- needs no schema change: `kind` is already `text` (ADR-0004's own note on
-- why applies here too), only application code that knows what to do with
-- the new value.

do $$ begin
  create type public.drift_status as enum ('detected', 'diagnosing', 'proposed', 'resolved', 'dismissed');
exception when duplicate_object then null;
end $$;

create table if not exists public.data_source_schedules (
  id                  uuid primary key default gen_random_uuid(),
  org_id              uuid not null references public.organizations (id) on delete cascade,
  source_id           uuid not null references public.data_sources (id) on delete cascade,
  cron                text not null default '0 6 * * *',  -- daily, 06:00 UTC
  enabled             boolean not null default true,
  -- ADR-0008 §4's opt-in, off by default: even the narrowest auto-apply case
  -- (additive schema drift, shadow run confirms it) requires this explicitly
  -- true. A schedule existing at all never implies consent to unattended writes.
  auto_apply_enabled  boolean not null default false,
  last_run_at         timestamptz,
  next_run_at         timestamptz not null default now(),
  created_by          uuid references auth.users (id) on delete set null,
  created_at          timestamptz not null default now(),
  unique (source_id)
);

-- The worker's own query: "which schedules are due", filtered to enabled ones.
create index if not exists data_source_schedules_due_idx
  on public.data_source_schedules (next_run_at) where enabled;

alter table public.data_source_schedules enable row level security;
alter table public.data_source_schedules force row level security;

do $$ begin
  create policy data_source_schedules_org_isolation on public.data_source_schedules
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;

create table if not exists public.drift_events (
  id           uuid primary key default gen_random_uuid(),
  org_id       uuid not null references public.organizations (id) on delete cascade,
  source_id    uuid not null references public.data_sources (id) on delete cascade,
  schedule_id  uuid references public.data_source_schedules (id) on delete set null,
  kind         text not null,  -- schema_change | statistical_shift — see lumen.sentinel.drift
  severity     real not null check (severity >= 0 and severity <= 1),
  details      jsonb not null default '{}'::jsonb,
  status       public.drift_status not null default 'detected',
  -- Convenience pointer to whatever pipeline_patch proposal this event
  -- produced, if any. Not the only way to find it — the proposal's own
  -- `spec->>'drift_event_id'` (ADR-0008 §3) is the record of truth — but a
  -- direct column is a plain index instead of a JSON containment query for
  -- every render of the drift feed.
  proposal_id  uuid references public.proposals (id) on delete set null,
  occurred_at  timestamptz not null default now()
);

create index if not exists drift_events_org_idx on public.drift_events (org_id, occurred_at desc);
create index if not exists drift_events_source_idx on public.drift_events (source_id, occurred_at desc);

alter table public.drift_events enable row level security;
alter table public.drift_events force row level security;

do $$ begin
  create policy drift_events_org_isolation on public.drift_events
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;
