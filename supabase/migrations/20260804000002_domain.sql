-- Lumen — the agent domain
--
-- Every table here carries org_id and delegates isolation to
-- public.is_org_member(). The proposal is the centre of the model: an agent
-- never mutates tenant data, it emits a spec that a human accepts or rejects,
-- and only an accepted spec is executed by the worker.

set check_function_bodies = off;

-- ────────────────────────────────────────────────────────────────────────────
-- Data sources
-- ────────────────────────────────────────────────────────────────────────────

do $$ begin
  create type public.source_kind   as enum ('postgres', 'mysql', 'csv', 'json', 'parquet');
  create type public.source_status as enum ('idle', 'syncing', 'live', 'error');
exception when duplicate_object then null;
end $$;

create table if not exists public.data_sources (
  id            uuid primary key default gen_random_uuid(),
  org_id        uuid not null references public.organizations (id) on delete cascade,
  name          text not null,
  kind          public.source_kind   not null,
  status        public.source_status not null default 'idle',
  -- Never returned by any read endpoint. Encrypted before it is written.
  dsn_encrypted text,
  object_path   text,          -- Supabase Storage path, e.g. org/<uuid>/uploads/<uuid>.csv
  table_name    text,
  row_count     bigint,
  byte_size     bigint,
  created_by    uuid references auth.users (id) on delete set null,
  created_at    timestamptz not null default now()
);

create index if not exists data_sources_org_idx on public.data_sources (org_id, created_at desc);

-- ────────────────────────────────────────────────────────────────────────────
-- Dataset handles — durable references to materialised Parquet
-- ────────────────────────────────────────────────────────────────────────────

create table if not exists public.dataset_handles (
  rid          text primary key,
  org_id       uuid not null references public.organizations (id) on delete cascade,
  source_id    uuid references public.data_sources (id) on delete set null,
  object_path  text   not null,
  backend      text   not null check (backend in ('pandas', 'polars', 'spark')),
  schema       jsonb  not null default '{}'::jsonb,
  row_count    bigint not null default 0,
  byte_size    bigint not null default 0,
  label        text   not null default '',
  created_at   timestamptz not null default now(),
  expires_at   timestamptz
);

create index if not exists dataset_handles_org_idx on public.dataset_handles (org_id, created_at desc);

-- ────────────────────────────────────────────────────────────────────────────
-- Runs, proposals, events
-- ────────────────────────────────────────────────────────────────────────────

do $$ begin
  create type public.run_status      as enum ('queued', 'running', 'succeeded', 'failed', 'cancelled');
  create type public.proposal_status as enum ('draft', 'awaiting_review', 'accepted', 'rejected', 'applied', 'failed');
  create type public.event_type      as enum ('thinking', 'tool_call', 'tool_result', 'message', 'proposal', 'error', 'done');
exception when duplicate_object then null;
end $$;

create table if not exists public.runs (
  id          uuid primary key default gen_random_uuid(),
  org_id      uuid not null references public.organizations (id) on delete cascade,
  source_id   uuid references public.data_sources (id) on delete set null,
  thread_id   uuid not null,
  kind        text not null,
  status      public.run_status not null default 'queued',
  backend     text not null default 'polars',
  created_by  uuid references auth.users (id) on delete set null,
  started_at  timestamptz not null default now(),
  finished_at timestamptz,
  error       text
);

create index if not exists runs_org_idx    on public.runs (org_id, started_at desc);
create index if not exists runs_thread_idx on public.runs (thread_id, started_at);

create table if not exists public.proposals (
  id             uuid primary key default gen_random_uuid(),
  org_id         uuid not null references public.organizations (id) on delete cascade,
  run_id         uuid not null references public.runs (id) on delete cascade,
  thread_id      uuid not null,
  author_agent   text not null,
  kind           text not null,
  status         public.proposal_status not null default 'awaiting_review',
  -- The executable plan, in exactly the shape PipelineBuilder.build() consumes.
  spec           jsonb not null,
  rationale      text  not null default '',
  estimate       jsonb not null default '{}'::jsonb,
  decided_by     uuid references auth.users (id) on delete set null,
  decided_at     timestamptz,
  applied_run_id uuid references public.runs (id) on delete set null,
  created_at     timestamptz not null default now()
);

create index if not exists proposals_org_idx    on public.proposals (org_id, created_at desc);
create index if not exists proposals_thread_idx on public.proposals (thread_id, created_at);

create table if not exists public.agent_events (
  id         uuid primary key default gen_random_uuid(),
  org_id     uuid not null references public.organizations (id) on delete cascade,
  run_id     uuid not null references public.runs (id) on delete cascade,
  seq        integer not null,
  type       public.event_type not null,
  payload    jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (run_id, seq)
);

create index if not exists agent_events_run_idx on public.agent_events (run_id, seq);

-- Sequence allocation lives in the database so two workers cannot collide.
create or replace function public.append_agent_event(
  p_org_id  uuid,
  p_run_id  uuid,
  p_type    public.event_type,
  p_payload jsonb
)
returns public.agent_events
language plpgsql
volatile
security invoker
set search_path = public
as $$
declare
  inserted public.agent_events;
begin
  insert into public.agent_events (org_id, run_id, seq, type, payload)
  select p_org_id,
         p_run_id,
         coalesce(max(e.seq), 0) + 1,
         p_type,
         coalesce(p_payload, '{}'::jsonb)
  from public.agent_events e
  where e.run_id = p_run_id
  returning * into inserted;

  return inserted;
end;
$$;

grant execute on function public.append_agent_event(uuid, uuid, public.event_type, jsonb)
  to authenticated, service_role;

-- ────────────────────────────────────────────────────────────────────────────
-- Usage — written in the same transaction as the work it measures
-- ────────────────────────────────────────────────────────────────────────────

create table if not exists public.usage_records (
  id          uuid primary key default gen_random_uuid(),
  org_id      uuid not null references public.organizations (id) on delete cascade,
  run_id      uuid references public.runs (id) on delete set null,
  agent       text,
  metric      text    not null,
  quantity    numeric(20, 4) not null,
  cost_micros bigint  not null default 0,
  occurred_at timestamptz not null default now(),
  meta        jsonb   not null default '{}'::jsonb
);

create index if not exists usage_records_org_idx on public.usage_records (org_id, occurred_at desc);

-- ────────────────────────────────────────────────────────────────────────────
-- Row-level security
--
-- One loop, so a table added later without a policy is a visible omission
-- rather than a silent leak. test_rls.py asserts this list matches reality.
-- ────────────────────────────────────────────────────────────────────────────

do $$
declare
  t text;
begin
  foreach t in array array[
    'data_sources', 'dataset_handles', 'runs', 'proposals', 'agent_events', 'usage_records'
  ]
  loop
    execute format('alter table public.%I enable row level security', t);
    execute format('alter table public.%I force row level security', t);
    execute format('drop policy if exists %I on public.%I', t || '_org_isolation', t);
    execute format(
      'create policy %I on public.%I using (public.is_org_member(org_id)) with check (public.is_org_member(org_id))',
      t || '_org_isolation', t
    );
  end loop;
end $$;
