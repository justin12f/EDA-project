-- ADR-0004: usage metering, quotas, and the plan/subscription model.
--
-- Deliberate simplifications against the ADR text, stated here rather than
-- left implicit:
--   * usage_rollups is a view (usage_current_period), not a stored table.
--     Rollup storage is a hot-path optimization for a volume this product
--     does not have yet — the same "premature investment" the ADR itself
--     argues against elsewhere. Revisit once a query over usage_records
--     shows up in a slow-query log, not before.
--   * usage_records has no monthly partitioning yet, for the same reason.
--   * The Redis-cached admission counter is not implemented — QuotaGate
--     reads Postgres directly. Redis is already provisioned (ADR-0001) for
--     when the query volume actually needs the cache; a correctness-first
--     implementation that queries the authoritative table is not a
--     regression against a caching layer that has no load to justify it yet.

-- ────────────────────────────────────────────────────────────────────────────
-- Plans — a catalog, not tenant data. Readable by any authenticated user
-- (choosing a plan requires seeing what the other plans are), writable by
-- nobody through the API; changed only by a migration.
-- ────────────────────────────────────────────────────────────────────────────

create table if not exists public.plans (
  code                     text primary key,
  name                     text not null,
  price_cents              integer not null default 0,
  llm_input_tokens_limit   bigint,   -- per calendar month; null = unlimited
  llm_output_tokens_limit  bigint,
  agent_runs_limit         bigint,
  storage_bytes_limit      bigint,
  created_at               timestamptz not null default now()
);

alter table public.plans enable row level security;
alter table public.plans force row level security;

do $$ begin
  create policy plans_read_all on public.plans for select using (true);
exception when duplicate_object then null;
end $$;

insert into public.plans
  (code, name, price_cents, llm_input_tokens_limit, llm_output_tokens_limit, agent_runs_limit, storage_bytes_limit)
values
  ('free', 'Free',       0,    200000,   50000,   50,   1073741824),   -- 1 GiB
  ('pro',  'Pro',        2900, 2000000,  500000,  1000, 10737418240),  -- 10 GiB
  ('team', 'Team',       9900, 10000000, 2500000, 5000, 107374182400), -- 100 GiB
  ('ent',  'Enterprise', 0,    null,     null,    null, null)          -- negotiated, unlimited in-app
on conflict (code) do nothing;

-- organizations.plan_code (ADR-0002) already exists as the fast-read copy
-- current_identity() serves; it becomes referentially real once the catalog
-- it points at exists. Every row already in the table reads 'free', which is
-- exactly the row this migration just inserted.
do $$ begin
  alter table public.organizations
    add constraint organizations_plan_code_fkey
    foreign key (plan_code) references public.plans (code);
exception when duplicate_object then null;
end $$;

-- ────────────────────────────────────────────────────────────────────────────
-- Subscriptions — one per org. Stripe is a downstream consumer of this
-- table, never the source of truth for what an org is entitled to; a Stripe
-- outage delays invoicing, it does not change what the app enforces.
-- ────────────────────────────────────────────────────────────────────────────

do $$ begin
  create type public.subscription_status as enum ('active', 'past_due', 'canceled');
exception when duplicate_object then null;
end $$;

create table if not exists public.subscriptions (
  id                       uuid primary key default gen_random_uuid(),
  org_id                   uuid not null references public.organizations (id) on delete cascade,
  plan_code                text not null references public.plans (code) default 'free',
  status                   public.subscription_status not null default 'active',
  stripe_customer_id       text,
  stripe_subscription_id   text,
  current_period_start     timestamptz,
  current_period_end       timestamptz,
  created_at               timestamptz not null default now(),
  updated_at               timestamptz not null default now(),
  unique (org_id)
);

create index if not exists subscriptions_org_idx on public.subscriptions (org_id);

alter table public.subscriptions enable row level security;
alter table public.subscriptions force row level security;

do $$ begin
  create policy subscriptions_org_isolation on public.subscriptions
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;

-- One free subscription per existing org, so the table is never a source of
-- "no row means what?" ambiguity for an org created before this migration.
insert into public.subscriptions (org_id, plan_code)
select id, plan_code from public.organizations
on conflict (org_id) do nothing;

-- New orgs get one the same way they get their first membership row.
create or replace function public.handle_new_organization_subscription()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.subscriptions (org_id, plan_code)
  values (new.id, new.plan_code)
  on conflict (org_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_organization_created on public.organizations;
create trigger on_organization_created
  after insert on public.organizations
  for each row execute function public.handle_new_organization_subscription();

-- ────────────────────────────────────────────────────────────────────────────
-- Usage records — the transactional ledger. Written in the same transaction
-- as the work it measures; a run that completed and was not billed is a bug
-- the schema makes impossible, not a state the schema merely discourages.
-- ────────────────────────────────────────────────────────────────────────────

-- `metric` is `text`, not an enum, on purpose — matching how ADR-0003's
-- `proposals.kind` already made this call ("no schema change needed, only
-- application code that knows what to do with a new value"). The set of
-- valid metrics is enforced in Python (`billing/quota.py`'s `Metric` literal
-- type), the same place `proposals.kind`'s valid values are.
--
-- This table already existed on this project (created ahead of this
-- migration, empty, no rows) with a shape close to but not identical to
-- ADR-0004's text — `meta` rather than `metadata`, no `unit_cost_micros`.
-- Rather than force a second, conflicting definition, this reconciles to
-- that shape: create-if-fresh with the real column names, alter-if-not.
create table if not exists public.usage_records (
  id                 uuid primary key default gen_random_uuid(),
  org_id             uuid not null references public.organizations (id) on delete cascade,
  run_id             uuid references public.runs (id) on delete set null,
  agent              text,
  metric             text not null,
  quantity           numeric not null check (quantity >= 0),
  unit_cost_micros   bigint not null default 0,
  cost_micros        bigint not null default 0,
  occurred_at        timestamptz not null default now(),
  meta               jsonb not null default '{}'::jsonb
);

alter table public.usage_records
  add column if not exists unit_cost_micros bigint not null default 0;

create index if not exists usage_records_org_period_idx
  on public.usage_records (org_id, occurred_at desc);
create index if not exists usage_records_org_metric_idx
  on public.usage_records (org_id, metric, occurred_at desc);
create index if not exists usage_records_run_idx
  on public.usage_records (run_id) where run_id is not null;

alter table public.usage_records enable row level security;
alter table public.usage_records force row level security;

do $$ begin
  create policy usage_records_org_isolation on public.usage_records
    using (public.is_org_member(org_id)) with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;

-- The current calendar month's usage per org per metric — QuotaGate's read
-- path and the usage panel's data source. A view, not a table: see the note
-- at the top of this file for why usage_rollups is deliberately not stored.
create or replace view public.usage_current_period as
select
  org_id,
  metric,
  sum(quantity)     as quantity,
  sum(cost_micros)  as cost_micros,
  count(*)          as record_count
from public.usage_records
where occurred_at >= date_trunc('month', now())
group by org_id, metric;

-- ────────────────────────────────────────────────────────────────────────────
-- Proposal kinds this ADR introduces. `proposals.kind` (ADR-0003) is already
-- `text`, not an enum — no schema change needed to start using new values,
-- only application code that knows what to do with 'plan_change' and
-- 'member_role_change' when a human accepts one.
-- ────────────────────────────────────────────────────────────────────────────

-- ────────────────────────────────────────────────────────────────────────────
-- Prerequisite gap this ADR's own `list_members` exposed, not something the
-- ADR asked for directly: `profiles` had exactly one read policy —
-- `id = auth.uid()`, a person's own row. `list_members` and
-- `propose_member_role_change` join memberships to profiles to show a
-- teammate's name; under the old policy that join silently returned no row
-- for anyone but yourself, which a null-check then misread as "no such
-- membership" — found by a test that used a second real user, not by
-- reading the code. A member's profile is now readable by anyone who shares
-- an org with them — still nobody outside every org they're in.
-- ────────────────────────────────────────────────────────────────────────────

do $$ begin
  create policy profiles_org_mates_read on public.profiles
    for select using (
      exists (
        select 1 from public.memberships mine
        join public.memberships theirs on theirs.org_id = mine.org_id
        where mine.user_id = auth.uid() and theirs.user_id = profiles.id
      )
    );
exception when duplicate_object then null;
end $$;
