-- Lumen — the agent's memory
--
-- Everything the agent learns about a dataset is stored here with an embedding:
-- profiles, schema summaries, the reasoning behind a proposal, and the decisions
-- humans made. On the second visit to a source the agent retrieves what it
-- concluded the first time — including which proposals were rejected and why.
--
-- 384 dimensions = BAAI/bge-small-en-v1.5, run locally via fastembed. No API key
-- is involved in embedding, which is what keeps the whole product runnable on a
-- fresh checkout with no credentials.

create extension if not exists vector with schema extensions;

set check_function_bodies = off;

do $$ begin
  create type public.context_kind as enum ('profile', 'schema', 'rationale', 'decision', 'note');
exception when duplicate_object then null;
end $$;

create table if not exists public.data_contexts (
  id         uuid primary key default gen_random_uuid(),
  org_id     uuid not null references public.organizations (id) on delete cascade,
  source_id  uuid references public.data_sources (id) on delete cascade,
  run_id     uuid references public.runs (id) on delete set null,
  rid        text,
  kind       public.context_kind not null,
  title      text not null default '',
  -- The exact text that was embedded. Kept so a re-embed after a model change
  -- needs no recomputation of the upstream analysis.
  content    text not null,
  embedding  extensions.vector(384),
  metadata   jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

comment on column public.data_contexts.embedding is
  'BAAI/bge-small-en-v1.5, 384d, cosine. Null until the embedder has run — search skips null rows.';

create index if not exists data_contexts_org_idx    on public.data_contexts (org_id, created_at desc);
create index if not exists data_contexts_source_idx on public.data_contexts (source_id, kind);

-- HNSW over cosine distance. m/ef_construction are the pgvector defaults, which
-- are the right starting point at this volume; tune only with a recall problem
-- in hand.
create index if not exists data_contexts_embedding_idx
  on public.data_contexts
  using hnsw (embedding extensions.vector_cosine_ops);

alter table public.data_contexts enable row level security;
alter table public.data_contexts force row level security;

drop policy if exists data_contexts_org_isolation on public.data_contexts;
create policy data_contexts_org_isolation on public.data_contexts
  using (public.is_org_member(org_id))
  with check (public.is_org_member(org_id));

-- ────────────────────────────────────────────────────────────────────────────
-- Semantic search
--
-- security invoker, so RLS still applies: a caller can only ever match rows in
-- an org they belong to. The explicit org filter is a second gate, not the
-- first — it also lets the planner use the org index before the vector index.
-- ────────────────────────────────────────────────────────────────────────────

create or replace function public.match_data_contexts(
  query_embedding extensions.vector(384),
  match_org       uuid,
  match_count     integer default 5,
  min_similarity  double precision default 0.25,
  filter_source   uuid default null,
  filter_kinds    public.context_kind[] default null
)
returns table (
  id         uuid,
  source_id  uuid,
  rid        text,
  kind       public.context_kind,
  title      text,
  content    text,
  metadata   jsonb,
  similarity double precision,
  created_at timestamptz
)
language sql
stable
security invoker
set search_path = public, extensions
as $$
  select c.id,
         c.source_id,
         c.rid,
         c.kind,
         c.title,
         c.content,
         c.metadata,
         1 - (c.embedding <=> query_embedding) as similarity,
         c.created_at
  from public.data_contexts c
  where c.org_id = match_org
    and c.embedding is not null
    and (filter_source is null or c.source_id = filter_source)
    and (filter_kinds  is null or c.kind = any(filter_kinds))
    and 1 - (c.embedding <=> query_embedding) >= min_similarity
  order by c.embedding <=> query_embedding
  limit greatest(1, least(match_count, 50));
$$;

grant execute on function public.match_data_contexts(
  extensions.vector(384), uuid, integer, double precision, uuid, public.context_kind[]
) to authenticated, service_role;

-- ────────────────────────────────────────────────────────────────────────────
-- Convenience: the most recent context of each kind for a source, without a
-- vector round trip. The agent asks this before it asks for a similarity search
-- — exact recall beats approximate recall when you know the key.
-- ────────────────────────────────────────────────────────────────────────────

create or replace function public.latest_source_contexts(
  match_org    uuid,
  match_source uuid
)
returns table (
  kind       public.context_kind,
  title      text,
  content    text,
  metadata   jsonb,
  created_at timestamptz
)
language sql
stable
security invoker
set search_path = public
as $$
  select distinct on (c.kind)
         c.kind, c.title, c.content, c.metadata, c.created_at
  from public.data_contexts c
  where c.org_id = match_org and c.source_id = match_source
  order by c.kind, c.created_at desc;
$$;

grant execute on function public.latest_source_contexts(uuid, uuid) to authenticated, service_role;
