-- ADR-0024: a connected customer database is mirrored before it is copied.
-- `discovered_structure` holds what information_schema reported, so the
-- diagram can render a table that exists at origin but has not been
-- imported — without creating an empty placeholder that would make our own
-- information_schema claim a table holding nothing.
alter table public.data_sources
  add column if not exists discovered_structure jsonb,
  add column if not exists imported_tables jsonb not null default '[]'::jsonb;
