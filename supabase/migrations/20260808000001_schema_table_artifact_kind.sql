-- ADR-0024: a table the Architect created is an artifact like a pipeline or
-- a canonical entity — something downstream work can depend on, and
-- something an impact simulation (ADR-0010) must be able to name.

alter type public.artifact_kind add value if not exists 'schema_table';
