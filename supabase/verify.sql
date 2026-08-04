-- Post-migration verification.
--
-- Two questions this answers that "Success. No rows returned" does not:
--   1. Is row-level security actually ON for every table carrying org_id?
--      The Supabase editor's static check could not see the RLS block because
--      it lives inside a DO $$ ... execute format(...) loop, so it has to be
--      confirmed against the catalogue rather than assumed.
--   2. Did pgvector install, and does the HNSW index exist?
--
-- Every row should read OK. Anything else is a defect to fix before writing
-- a single row of tenant data.

with expected as (
  select unnest(array[
    'profiles', 'organizations', 'memberships',
    'data_sources', 'dataset_handles', 'runs', 'proposals',
    'agent_events', 'usage_records', 'data_contexts'
  ]) as table_name
),
tables as (
  select e.table_name,
         c.oid is not null                       as exists,
         coalesce(c.relrowsecurity, false)        as rls_enabled,
         coalesce(c.relforcerowsecurity, false)   as rls_forced,
         (select count(*) from pg_policy p where p.polrelid = c.oid) as policies
  from expected e
  left join pg_class c
         on c.relname = e.table_name
        and c.relnamespace = 'public'::regnamespace
        and c.relkind = 'r'
)
select 'table: ' || table_name as check,
       case
         when not exists                       then 'MISSING'
         when not rls_enabled                  then 'FAIL — RLS disabled'
         when policies = 0                     then 'FAIL — no policy'
         when not rls_forced and table_name <> 'profiles' then 'WARN — RLS not forced'
         else 'OK (' || policies || ' policies)'
       end as result
from tables

union all
select 'extension: vector',
       case when exists (select 1 from pg_extension where extname = 'vector')
            then 'OK' else 'MISSING' end

union all
select 'index: data_contexts hnsw',
       case when exists (
              select 1 from pg_indexes
              where schemaname = 'public' and indexname = 'data_contexts_embedding_idx')
            then 'OK' else 'MISSING' end

union all
select 'function: ' || fn,
       case when exists (
              select 1 from pg_proc p
              join pg_namespace n on n.oid = p.pronamespace
              where n.nspname = 'public' and p.proname = fn)
            then 'OK' else 'MISSING' end
from unnest(array[
  'is_org_member', 'has_org_role', 'handle_new_user',
  'current_identity', 'append_agent_event',
  'match_data_contexts', 'latest_source_contexts'
]) as fn

union all
select 'trigger: on_auth_user_created',
       case when exists (select 1 from pg_trigger where tgname = 'on_auth_user_created')
            then 'OK' else 'MISSING' end

order by 1;
