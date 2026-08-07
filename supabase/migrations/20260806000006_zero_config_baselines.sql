-- ADR-0013: zero-config anomaly baselines. `column_baselines` is keyed by
-- (source_id, column_name) because that pair is exactly what identifies
-- "this column, on this source" across scans — a rename shows up as a new
-- baseline starting cold, which is correct: a renamed column's history
-- does not obviously transfer (ADR-0008's rename detection is a heuristic
-- guess, not something a calibrated band should silently inherit from).
--
-- `override` is nullable and separate from `model`: `model` is what the
-- Sentinel calibrated on its own and keeps overwriting every scan;
-- `override` is what a person explicitly set and the Sentinel never
-- touches (Action Item 6 — "never required", so absence is the default,
-- not a sentinel value inside a jsonb blob the calibration logic could
-- accidentally clobber).
--
-- `source_baselines` gets the same treatment one level up — row-count
-- drift is a property of the source, not of any one column (Action Item
-- 4) — but has no column_name to key on, hence a separate table rather
-- than a fake pseudo-column inside column_baselines.

create table if not exists public.column_baselines (
  id            uuid primary key default gen_random_uuid(),
  org_id        uuid not null references public.organizations (id) on delete cascade,
  source_id     uuid not null references public.data_sources (id) on delete cascade,
  column_name   text not null,
  baseline_kind text not null,  -- numeric | categorical | datetime | freeform_string
  model         jsonb not null default '{}'::jsonb,
  sample_size   integer not null default 0,
  override      jsonb,
  computed_at   timestamptz not null default now(),
  created_at    timestamptz not null default now(),
  unique (source_id, column_name)
);

create index if not exists column_baselines_org_idx on public.column_baselines (org_id);

alter table public.column_baselines enable row level security;
alter table public.column_baselines force row level security;

do $$ begin
  create policy column_baselines_org_isolation on public.column_baselines
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;

create table if not exists public.source_baselines (
  id           uuid primary key default gen_random_uuid(),
  org_id       uuid not null references public.organizations (id) on delete cascade,
  source_id    uuid not null references public.data_sources (id) on delete cascade unique,
  model        jsonb not null default '{}'::jsonb,
  sample_size  integer not null default 0,
  computed_at  timestamptz not null default now(),
  created_at   timestamptz not null default now()
);

alter table public.source_baselines enable row level security;
alter table public.source_baselines force row level security;

do $$ begin
  create policy source_baselines_org_isolation on public.source_baselines
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;
