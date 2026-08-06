-- ADR-0009: the verified business glossary and the read-only trust API.
--
-- Entity resolution and certification reuse infrastructure ADR-0006/0008
-- already paid for: per-column embeddings piggyback on the existing
-- data_contexts table (Kind.SCHEMA was reserved, unused, for exactly this —
-- see context/store.py), and certification composes DriftEvent/Proposal
-- state on demand rather than becoming a second source of truth to keep in
-- sync. proposals.kind needs no change either — already text (ADR-0004's own
-- note on why, and ADR-0008's pipeline_patch after it).
--
-- api_keys is the one genuinely new thing here: the first credential in this
-- product that is not a Supabase-issued JWT. It gets its own table and its
-- own lookup path (by hash, before any org is known) rather than trying to
-- make Supabase auth issue non-human credentials — but the RLS *session* a
-- valid key opens for reading tenant data is the exact same user_session()
-- every other tenant read already uses, resolved to a live member of the
-- key's org. No second isolation mechanism; the existing one gets a second
-- way to be authenticated into, per the ADR's own framing.

do $$ begin
  create type public.canonical_entity_status as enum ('proposed', 'approved', 'deprecated');
exception when duplicate_object then null;
end $$;

create table if not exists public.canonical_entities (
  id                   uuid primary key default gen_random_uuid(),
  org_id               uuid not null references public.organizations (id) on delete cascade,
  name                 text not null,
  entity_type          text not null,
  reconciliation_rule  jsonb not null default '{}'::jsonb,
  status               public.canonical_entity_status not null default 'approved',
  -- Which entity_mapping proposal approved this — a convenience pointer, not
  -- the record of truth (the proposal's own spec is), same relationship
  -- drift_events.proposal_id has to a pipeline_patch (ADR-0008).
  proposal_id          uuid references public.proposals (id) on delete set null,
  created_by           uuid references auth.users (id) on delete set null,
  created_at           timestamptz not null default now(),
  -- Two entities sharing a name in one org is almost certainly the same
  -- concept proposed twice, not a coincidence — the clustering job's dedupe
  -- check (never re-propose a rejected or already-covered cluster) relies on
  -- this being enforced, not just conventional.
  unique (org_id, name)
);

create table if not exists public.canonical_entity_members (
  id          uuid primary key default gen_random_uuid(),
  entity_id   uuid not null references public.canonical_entities (id) on delete cascade,
  source_id   uuid not null references public.data_sources (id) on delete cascade,
  column_name text not null,
  created_at  timestamptz not null default now(),
  -- A column resolves to at most one business concept — that is the entire
  -- point of resolution. Two entities both claiming orders.customer_id is a
  -- conflict to surface, not a state to allow silently.
  unique (source_id, column_name)
);

create index if not exists canonical_entities_org_idx on public.canonical_entities (org_id);
create index if not exists canonical_entity_members_entity_idx on public.canonical_entity_members (entity_id);

alter table public.canonical_entities enable row level security;
alter table public.canonical_entities force row level security;
alter table public.canonical_entity_members enable row level security;
alter table public.canonical_entity_members force row level security;

do $$ begin
  create policy canonical_entities_org_isolation on public.canonical_entities
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;

-- canonical_entity_members has no org_id column of its own — scoped through
-- the parent entity, the same shape drift_events uses for schedule_id.
do $$ begin
  create policy canonical_entity_members_org_isolation on public.canonical_entity_members
    using (exists (
      select 1 from public.canonical_entities e
      where e.id = entity_id and public.is_org_member(e.org_id)
    ))
    with check (exists (
      select 1 from public.canonical_entities e
      where e.id = entity_id and public.is_org_member(e.org_id)
    ));
exception when duplicate_object then null;
end $$;

-- api_keys: the raw credential is shown once and never stored — only its
-- hash. `scope` is a single enumerated, read-only value per key, on purpose:
-- no scope value exists that grants a write (see the ADR's Option B risk
-- analysis on why this is the whole mitigation, not cleverness elsewhere).
do $$ begin
  create type public.api_key_scope as enum ('read:glossary', 'read:certification');
exception when duplicate_object then null;
end $$;

create table if not exists public.api_keys (
  id          uuid primary key default gen_random_uuid(),
  org_id      uuid not null references public.organizations (id) on delete cascade,
  name        text not null,
  key_hash    text not null,
  -- First 12 chars of the raw key, kept so a person can tell two keys apart
  -- in a list without the product ever being able to show the secret again.
  key_prefix  text not null,
  scope       public.api_key_scope not null,
  created_by  uuid references auth.users (id) on delete set null,
  created_at  timestamptz not null default now(),
  revoked_at  timestamptz
);

-- The lookup a trust-API request actually takes: by hash, before any org or
-- user identity is known at all. This index is what that query hits —
-- authentication happens before there is any auth.uid() to scope RLS by,
-- which is the entire reason this table and this path exist.
create unique index if not exists api_keys_hash_idx on public.api_keys (key_hash);
create index if not exists api_keys_org_idx on public.api_keys (org_id);

alter table public.api_keys enable row level security;
alter table public.api_keys force row level security;

-- Management (issue / list / revoke) goes through a signed-in owner, same
-- reasoning as ADR-0004's plan_change: this changes what an outside party
-- can do, not just what the data looks like. The hash-lookup path used at
-- authentication time reads through service_session() instead of this
-- policy — there is no auth.uid() yet at that point.
do $$ begin
  create policy api_keys_owner_manage on public.api_keys
    for all
    using (public.has_org_role(org_id, array['owner']::public.org_role[]))
    with check (public.has_org_role(org_id, array['owner']::public.org_role[]));
exception when duplicate_object then null;
end $$;
