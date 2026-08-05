-- Lumen — identity and tenancy
--
-- Supabase Auth (auth.users) owns credentials. This migration owns everything
-- above that: a profile mirror we can join to, organizations, memberships, and
-- the one SECURITY DEFINER helper every RLS policy in the system calls.
--
-- Isolation invariant: no table carrying org_id may exist without RLS enabled
-- and a policy delegating to public.is_org_member(). A test enumerates pg_class
-- and fails the build if one appears.

set check_function_bodies = off;

-- ────────────────────────────────────────────────────────────────────────────
-- Profiles — the joinable mirror of auth.users
-- ────────────────────────────────────────────────────────────────────────────

create table if not exists public.profiles (
  id            uuid primary key references auth.users (id) on delete cascade,
  email         text        not null,
  display_name  text        not null default '',
  avatar_url    text,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

comment on table public.profiles is
  'Mirror of auth.users. Kept in sync by handle_new_user(); never written directly by the app.';

alter table public.profiles enable row level security;

create policy profiles_self_read on public.profiles
  for select using (id = auth.uid());

create policy profiles_self_update on public.profiles
  for update using (id = auth.uid()) with check (id = auth.uid());

-- ────────────────────────────────────────────────────────────────────────────
-- Organizations and memberships
-- ────────────────────────────────────────────────────────────────────────────

create table if not exists public.organizations (
  id          uuid primary key default gen_random_uuid(),
  name        text        not null,
  slug        text        not null unique,
  plan_code   text        not null default 'free',
  created_by  uuid        references auth.users (id) on delete set null,
  created_at  timestamptz not null default now()
);

alter table public.organizations enable row level security;

do $$ begin
  create type public.org_role as enum ('owner', 'admin', 'member', 'viewer');
exception when duplicate_object then null;
end $$;

create table if not exists public.memberships (
  id         uuid primary key default gen_random_uuid(),
  org_id     uuid           not null references public.organizations (id) on delete cascade,
  user_id    uuid           not null references auth.users (id) on delete cascade,
  role       public.org_role not null default 'member',
  created_at timestamptz    not null default now(),
  unique (org_id, user_id)
);

create index if not exists memberships_user_idx on public.memberships (user_id);
create index if not exists memberships_org_idx  on public.memberships (org_id);

alter table public.memberships enable row level security;

-- ────────────────────────────────────────────────────────────────────────────
-- The isolation helpers
--
-- SECURITY DEFINER matters here: a policy on memberships that queried
-- memberships would recurse. Defining the check as a definer-rights function
-- breaks the cycle, and `set search_path` keeps it from being hijacked.
-- ────────────────────────────────────────────────────────────────────────────

create or replace function public.is_org_member(target uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.memberships m
    where m.org_id = target and m.user_id = auth.uid()
  );
$$;

create or replace function public.has_org_role(target uuid, allowed public.org_role[])
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.memberships m
    where m.org_id = target
      and m.user_id = auth.uid()
      and m.role = any(allowed)
  );
$$;

revoke all on function public.is_org_member(uuid) from public;
revoke all on function public.has_org_role(uuid, public.org_role[]) from public;
grant execute on function public.is_org_member(uuid) to authenticated, service_role;
grant execute on function public.has_org_role(uuid, public.org_role[]) to authenticated, service_role;

-- Policies that depend on the helpers
create policy organizations_member_read on public.organizations
  for select using (public.is_org_member(id));

create policy organizations_admin_update on public.organizations
  for update using (public.has_org_role(id, array['owner', 'admin']::public.org_role[]))
  with check  (public.has_org_role(id, array['owner', 'admin']::public.org_role[]));

create policy memberships_member_read on public.memberships
  for select using (public.is_org_member(org_id));

create policy memberships_admin_write on public.memberships
  for all using (public.has_org_role(org_id, array['owner', 'admin']::public.org_role[]))
  with check  (public.has_org_role(org_id, array['owner', 'admin']::public.org_role[]));

-- ────────────────────────────────────────────────────────────────────────────
-- Bootstrap: every new auth user gets a profile and a personal organization
--
-- Idempotent on purpose. Sign-up, OAuth first sign-in and a manual backfill can
-- all reach this path, and none of them may create a second org for the same
-- user.
-- ────────────────────────────────────────────────────────────────────────────

create or replace function public.slugify(value text)
returns text
language sql
immutable
as $$
  select trim(both '-' from regexp_replace(lower(coalesce(value, '')), '[^a-z0-9]+', '-', 'g'));
$$;

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  resolved_name text;
  base_slug     text;
  new_org       uuid;
begin
  resolved_name := coalesce(
    nullif(new.raw_user_meta_data ->> 'display_name', ''),
    nullif(new.raw_user_meta_data ->> 'full_name', ''),
    nullif(new.raw_user_meta_data ->> 'name', ''),
    split_part(new.email, '@', 1)
  );

  insert into public.profiles (id, email, display_name, avatar_url)
  values (new.id, new.email, resolved_name, new.raw_user_meta_data ->> 'avatar_url')
  on conflict (id) do update
    set email = excluded.email,
        updated_at = now();

  -- One personal organization per user, created only if they have none.
  if not exists (select 1 from public.memberships where user_id = new.id) then
    base_slug := coalesce(nullif(public.slugify(resolved_name), ''), 'workspace');

    insert into public.organizations (name, slug, plan_code, created_by)
    values (
      coalesce(nullif(new.raw_user_meta_data ->> 'org_name', ''), resolved_name || '''s workspace'),
      base_slug || '-' || substr(replace(new.id::text, '-', ''), 1, 6),
      'free',
      new.id
    )
    returning id into new_org;

    insert into public.memberships (org_id, user_id, role)
    values (new_org, new.id, 'owner');
  end if;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ────────────────────────────────────────────────────────────────────────────
-- What the signed-in user can see about themselves in one round trip
-- ────────────────────────────────────────────────────────────────────────────

create or replace function public.current_identity()
returns table (
  user_id      uuid,
  email        text,
  display_name text,
  avatar_url   text,
  org_id       uuid,
  org_name     text,
  org_slug     text,
  plan_code    text,
  role         public.org_role
)
language sql
stable
security invoker
set search_path = public
as $$
  select p.id, p.email, p.display_name, p.avatar_url,
         o.id, o.name, o.slug, o.plan_code, m.role
  from public.profiles p
  join public.memberships m on m.user_id = p.id
  join public.organizations o on o.id = m.org_id
  where p.id = auth.uid()
  order by m.created_at
  limit 1;
$$;

grant execute on function public.current_identity() to authenticated;
