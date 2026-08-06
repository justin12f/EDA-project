-- ADR-0009 §4/action item 8: an audit log for every trust-API request,
-- called out explicitly as "a security review surface, not optional" — the
-- first externally-callable surface in this product, so what it was asked
-- and by which key has to be reconstructable after the fact, not just rate
-- limited in the moment.

create table if not exists public.api_key_audit_log (
  id           uuid primary key default gen_random_uuid(),
  api_key_id   uuid references public.api_keys (id) on delete set null,
  org_id       uuid not null references public.organizations (id) on delete cascade,
  endpoint     text not null,
  status_code  integer not null,
  requested_at timestamptz not null default now()
);

-- The trust API's own rate limit reads this table (count this key's rows in
-- the trailing window) before deciding whether to serve the request — the
-- index this needs and the index a person reviewing "what has this key been
-- doing" needs are the same one.
create index if not exists api_key_audit_log_key_idx on public.api_key_audit_log (api_key_id, requested_at desc);
create index if not exists api_key_audit_log_org_idx on public.api_key_audit_log (org_id, requested_at desc);

alter table public.api_key_audit_log enable row level security;
alter table public.api_key_audit_log force row level security;

-- Read: owners reviewing their own org's trail, same bar as api_keys itself.
-- Write: any resolved org member, because the writer is always the trust
-- API acting as the key's resolved representative member
-- (`api_keys.authenticate`), never a person directly — there is no "this
-- person decided to log something" action to gate more tightly than that.
do $$ begin
  create policy api_key_audit_log_owner_read on public.api_key_audit_log
    for select using (public.has_org_role(org_id, array['owner']::public.org_role[]));
exception when duplicate_object then null;
end $$;

do $$ begin
  create policy api_key_audit_log_member_write on public.api_key_audit_log
    for insert with check (public.is_org_member(org_id));
exception when duplicate_object then null;
end $$;
