"""A live customer Postgres.

The reason D10 mirrors structure before copying a byte: this adapter does
not infer anything. Types, primary keys and foreign keys are read from the
customer's own information_schema, so the diagram shows constraints the
database actually holds rather than relationships guessed from containment.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import polars as pl

from lumen.architect.adapters.base import DiscoveredColumn, DiscoveredStructure, DiscoveredTable
from lumen.architect.spec import SqlType

# Postgres's own type names onto our closed enum. Anything absent falls to
# TEXT, which is lossless for display and honest about what we understand.
_TYPES: dict[str, SqlType] = {
    "smallint": SqlType.INTEGER,
    "integer": SqlType.INTEGER,
    "bigint": SqlType.BIGINT,
    "numeric": SqlType.NUMERIC,
    "real": SqlType.DOUBLE,
    "double precision": SqlType.DOUBLE,
    "boolean": SqlType.BOOLEAN,
    "text": SqlType.TEXT,
    "character varying": SqlType.VARCHAR,
    "character": SqlType.VARCHAR,
    "date": SqlType.DATE,
    "timestamp without time zone": SqlType.TIMESTAMP,
    "timestamp with time zone": SqlType.TIMESTAMPTZ,
    "uuid": SqlType.UUID,
    "json": SqlType.JSONB,
    "jsonb": SqlType.JSONB,
}

_COLUMNS = """
select table_name, column_name, data_type, is_nullable,
       character_maximum_length, numeric_precision, numeric_scale
from information_schema.columns
where table_schema = $1
order by table_name, ordinal_position
"""

_PRIMARY_KEYS = """
select tc.table_name, kcu.column_name
from information_schema.table_constraints tc
join information_schema.key_column_usage kcu
  on kcu.constraint_name = tc.constraint_name
 and kcu.table_schema = tc.table_schema
where tc.table_schema = $1 and tc.constraint_type = 'PRIMARY KEY'
order by kcu.ordinal_position
"""

_FOREIGN_KEYS = """
select tc.table_name        as from_table,
       kcu.column_name      as from_column,
       ccu.table_name       as to_table,
       ccu.column_name      as to_column
from information_schema.table_constraints tc
join information_schema.key_column_usage kcu
  on kcu.constraint_name = tc.constraint_name
 and kcu.table_schema = tc.table_schema
join information_schema.constraint_column_usage ccu
  on ccu.constraint_name = tc.constraint_name
 and ccu.table_schema = tc.table_schema
where tc.table_schema = $1 and tc.constraint_type = 'FOREIGN KEY'
"""


def _type_of(row: Any) -> tuple[SqlType, str | None]:
    sql_type = _TYPES.get(row["data_type"], SqlType.TEXT)
    if sql_type is SqlType.VARCHAR and row["character_maximum_length"]:
        return sql_type, str(row["character_maximum_length"])
    if sql_type is SqlType.NUMERIC and row["numeric_precision"]:
        return sql_type, f"{row['numeric_precision']},{row['numeric_scale'] or 0}"
    return sql_type, None


class PostgresAdapter:
    kind = "postgres"
    supports_incremental = True

    def __init__(self, dsn: str, schema: str = "public") -> None:
        self._dsn = dsn
        self._schema = schema
        self._known: set[str] = set()

    async def discover(self) -> DiscoveredStructure:
        conn = await asyncpg.connect(self._dsn)
        try:
            columns = await conn.fetch(_COLUMNS, self._schema)
            primary = await conn.fetch(_PRIMARY_KEYS, self._schema)
            foreign = await conn.fetch(_FOREIGN_KEYS, self._schema)
        finally:
            await conn.close()

        by_table: dict[str, list[DiscoveredColumn]] = {}
        for row in columns:
            sql_type, type_arg = _type_of(row)
            by_table.setdefault(row["table_name"], []).append(
                DiscoveredColumn(
                    name=row["column_name"],
                    sql_type=sql_type,
                    type_arg=type_arg,
                    nullable=row["is_nullable"] == "YES",
                )
            )

        keys: dict[str, list[str]] = {}
        for row in primary:
            keys.setdefault(row["table_name"], []).append(row["column_name"])

        references: dict[str, list[tuple[str, str, str]]] = {}
        for row in foreign:
            references.setdefault(row["from_table"], []).append(
                (row["from_column"], row["to_table"], row["to_column"])
            )

        tables = tuple(
            DiscoveredTable(
                name=name,
                columns=tuple(cols),
                primary_key=tuple(keys[name]) if name in keys else None,
                foreign_keys=tuple(references.get(name, ())),
            )
            for name, cols in sorted(by_table.items())
        )
        self._known = {t.name for t in tables}
        return DiscoveredStructure(tables=tables, declared=True)

    async def read(self, table: str, limit: int | None = None) -> Any:
        # A table name cannot be a bind parameter, so it is interpolated —
        # which makes this check the only thing between the adapter and
        # injection. The name must be one discovery already returned; it is
        # never taken from user or model input.
        if not self._known:
            await self.discover()
        if table not in self._known:
            raise ValueError(f"table {table!r} was not discovered on this source")

        suffix = f" LIMIT {int(limit)}" if limit is not None else ""
        query = f'SELECT * FROM "{self._schema}"."{table}"{suffix}'
        conn = await asyncpg.connect(self._dsn)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
        return pl.DataFrame([dict(row) for row in rows])
