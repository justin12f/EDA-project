-- ADR-0010: impact simulation before approval.
--
-- artifact_dependencies is a registry, not a graph engine — "who reads
-- what," written transactionally by the code that already knows the
-- dependency at creation time (apply_pipeline.py, proposals.py's
-- entity_mapping accept), never inferred later by parsing specs. Only two
-- of the three artifact kinds the ADR names have a real write path today:
-- `pipeline` (a source's currently-applied cleaning pipeline) and
-- `canonical_entity` (ADR-0009's cross-source mappings). `analysis_widget`
-- is kept in the enum because the ADR names it and it costs nothing to
-- reserve, but no code writes it — this product has no persisted
-- "dashboard widget" concept yet, confirmed before writing this migration
-- (run_statistic/compare_predictors/forecast_series all return straight to
-- the chat response, nothing is inserted anywhere). Building that
-- persistence layer is out of scope for this ADR; it was never asked for.

do $$ begin
  create type public.artifact_kind as enum ('pipeline', 'canonical_entity', 'analysis_widget');
exception when duplicate_object then null;
end $$;

create table if not exists public.artifact_dependencies (
  id           uuid primary key default gen_random_uuid(),
  org_id       uuid not null references public.organizations (id) on delete cascade,
  artifact_kind public.artifact_kind not null,
  -- Polymorphic on purpose (no FK): a `pipeline` artifact_id is the
  -- proposals.id whose apply made it current; a `canonical_entity` one is
  -- canonical_entities.id. One column, two possible tables — a lineage
  -- registry does not need a graph engine's referential integrity to be
  -- useful, only "who reads what, queried by source_id and columns."
  artifact_id  uuid not null,
  source_id    uuid not null references public.data_sources (id) on delete cascade,
  columns      text[] not null,
  created_at   timestamptz not null default now()
);

-- The one query this table exists to answer: "what currently reads this
-- source, on any of these columns" — a straight `&&` overlap, not a join.
create index if not exists artifact_dependencies_source_idx
  on public.artifact_dependencies (source_id);
create index if not exists artifact_dependencies_columns_gin_idx
  on public.artifact_dependencies using gin (columns);

alter table public.artifact_dependencies enable row level security;
alter table public.artifact_dependencies force row level security;

do $$ begin
  create policy artifact_dependencies_org_isolation on public.artifact_dependencies
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;

create table if not exists public.impact_reports (
  id                 uuid primary key default gen_random_uuid(),
  -- One report per proposal — computed once, synchronously, when the
  -- proposal is created (ADR-0010's Option D: never while a person is
  -- staring at a loading spinner on the review screen).
  proposal_id        uuid not null unique references public.proposals (id) on delete cascade,
  org_id             uuid not null references public.organizations (id) on delete cascade,
  dependents_checked int not null,
  -- Not in the ADR's own schema sketch, but its Action Item #6 and §4 both
  -- need it ("N of M dependents checked" when the cap is hit) — without the
  -- total, "10 checked" cannot honestly distinguish "10 of 10" from
  -- "10 of 340 truncated," which is exactly the false confidence §4 warns
  -- against.
  dependents_total   int not null,
  findings           jsonb not null default '[]'::jsonb,
  summary            text not null default '',
  computed_at        timestamptz not null default now()
);

alter table public.impact_reports enable row level security;
alter table public.impact_reports force row level security;

do $$ begin
  create policy impact_reports_org_isolation on public.impact_reports
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;
