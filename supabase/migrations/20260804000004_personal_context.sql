-- Lumen — per-user personalisation of the agent's memory
--
-- data_contexts was scoped to the organization only. That is right for facts
-- about shared data — a null rate is a property of the dataset, and every member
-- should benefit from it being measured once — and wrong for everything that
-- reflects a particular person's judgment.
--
-- Two scopes, decided by what the row *is*:
--
--   'org'   objective facts about shared data: profiles, schemas, applied
--           pipelines. Written once, useful to everyone, and re-deriving them
--           per user would mean paying for the same computation N times.
--
--   'user'  facts about one person: what they asked, what they rejected and why,
--           the phrasing they prefer, the columns they always care about. These
--           are private to their author even inside the same organization —
--           a teammate's rejected proposal is not evidence about your intent,
--           and treating it as such makes the agent worse for both of you.
--
-- The isolation is enforced in the policy, not in application code, so a
-- forgotten predicate cannot leak one person's history into another's context.

set check_function_bodies = off;

do $$ begin
  create type public.context_scope as enum ('org', 'user');
exception when duplicate_object then null;
end $$;

alter table public.data_contexts
  add column if not exists user_id uuid references auth.users (id) on delete cascade,
  add column if not exists scope public.context_scope not null default 'org';

-- A user-scoped row without an author would be invisible to everyone, which is
-- a silent data-loss bug rather than a safe default. Refuse it at write time.
alter table public.data_contexts
  drop constraint if exists data_contexts_user_scope_needs_author;
alter table public.data_contexts
  add constraint data_contexts_user_scope_needs_author
  check (scope = 'org' or user_id is not null);

create index if not exists data_contexts_user_idx
  on public.data_contexts (user_id, created_at desc)
  where user_id is not null;

-- ────────────────────────────────────────────────────────────────────────────
-- Visibility
--
-- Replaces the org-only policy. Reads: org-scoped rows for any member, plus your
-- own user-scoped rows. Writes: you may only author rows as yourself.
-- ────────────────────────────────────────────────────────────────────────────

drop policy if exists data_contexts_org_isolation on public.data_contexts;

create policy data_contexts_read on public.data_contexts
  for select using (
    public.is_org_member(org_id)
    and (scope = 'org' or user_id = auth.uid())
  );

create policy data_contexts_write on public.data_contexts
  for insert with check (
    public.is_org_member(org_id)
    and (scope = 'org' or user_id = auth.uid())
  );

create policy data_contexts_update on public.data_contexts
  for update using (
    public.is_org_member(org_id)
    and (scope = 'org' or user_id = auth.uid())
  )
  with check (
    public.is_org_member(org_id)
    and (scope = 'org' or user_id = auth.uid())
  );

create policy data_contexts_delete on public.data_contexts
  for delete using (
    public.is_org_member(org_id)
    and (scope = 'org' or user_id = auth.uid())
  );

-- ────────────────────────────────────────────────────────────────────────────
-- Search, personalised
--
-- security invoker, so the policies above do the filtering: a caller physically
-- cannot match another person's private context. `prefer_personal` biases the
-- ranking rather than the visibility — your own notes outrank a shared profile
-- of equal similarity, which is what makes a second visit feel like the agent
-- remembers *you* and not just the table.
-- ────────────────────────────────────────────────────────────────────────────

drop function if exists public.match_data_contexts(
  extensions.vector(384), uuid, integer, double precision, uuid, public.context_kind[]
);

create or replace function public.match_data_contexts(
  query_embedding extensions.vector(384),
  match_org       uuid,
  match_count     integer default 5,
  min_similarity  double precision default 0.25,
  filter_source   uuid default null,
  filter_kinds    public.context_kind[] default null,
  filter_scope    public.context_scope default null,
  prefer_personal double precision default 0.05
)
returns table (
  id         uuid,
  source_id  uuid,
  rid        text,
  kind       public.context_kind,
  scope      public.context_scope,
  user_id    uuid,
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
         c.scope,
         c.user_id,
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
    and (filter_scope  is null or c.scope = filter_scope)
    and 1 - (c.embedding <=> query_embedding) >= min_similarity
  order by
    (1 - (c.embedding <=> query_embedding))
      + case when c.user_id = auth.uid() then prefer_personal else 0 end
    desc
  limit greatest(1, least(match_count, 50));
$$;

grant execute on function public.match_data_contexts(
  extensions.vector(384), uuid, integer, double precision, uuid,
  public.context_kind[], public.context_scope, double precision
) to authenticated, service_role;

-- ────────────────────────────────────────────────────────────────────────────
-- What the agent loads before it starts: everything the org knows about this
-- source, plus everything this person in particular has decided about it.
-- ────────────────────────────────────────────────────────────────────────────

create or replace function public.personal_source_context(
  match_org    uuid,
  match_source uuid,
  max_personal integer default 10
)
returns table (
  scope      public.context_scope,
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
  (
    -- The most recent shared fact of each kind.
    select distinct on (c.kind) c.scope, c.kind, c.title, c.content, c.metadata, c.created_at
    from public.data_contexts c
    where c.org_id = match_org and c.source_id = match_source and c.scope = 'org'
    order by c.kind, c.created_at desc
  )
  union all
  (
    -- This person's own history, newest first.
    select c.scope, c.kind, c.title, c.content, c.metadata, c.created_at
    from public.data_contexts c
    where c.org_id = match_org
      and c.source_id = match_source
      and c.scope = 'user'
      and c.user_id = auth.uid()
    order by c.created_at desc
    limit greatest(1, least(max_personal, 50))
  );
$$;

grant execute on function public.personal_source_context(uuid, uuid, integer)
  to authenticated, service_role;
