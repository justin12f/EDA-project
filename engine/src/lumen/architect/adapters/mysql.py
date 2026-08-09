"""A live customer MySQL.

Structurally the same job as the Postgres adapter, but MySQL's
information_schema differs in three ways that matter: KEY_COLUMN_USAGE
carries REFERENCED_TABLE_NAME directly so there is no referential_
constraints join, the "schema" is the database name, and the type
vocabulary is its own.
"""

from __future__ import annotations

from typing import Any

import aiomysql
import polars as pl

from lumen.architect.adapters.base import DiscoveredColumn, DiscoveredStructure, DiscoveredTable
from lumen.architect.spec import SqlType

_TYPES: dict[str, SqlType] = {
    "tinyint": SqlType.INTEGER,
    "smallint": SqlType.INTEGER,
    "mediumint": SqlType.INTEGER,
    "int": SqlType.INTEGER,
    "integer": SqlType.INTEGER,
    "bigint": SqlType.BIGINT,
    "decimal": SqlType.NUMERIC,
    "numeric": SqlType.NUMERIC,
    "float": SqlType.DOUBLE,
    "double": SqlType.DOUBLE,
    "bit": SqlType.BOOLEAN,
    "char": SqlType.VARCHAR,
    "varchar": SqlType.VARCHAR,
    "text": SqlType.TEXT,
    "tinytext": SqlType.TEXT,
    "mediumtext": SqlType.TEXT,
    "longtext": SqlType.TEXT,
    "date": SqlType.DATE,
    "datetime": SqlType.TIMESTAMP,
    # MySQL converts a timestamp to UTC on store and back on read; datetime
    # is stored verbatim. They are genuinely different and collapsing them
    # loses the zone.
    "timestamp": SqlType.TIMESTAMPTZ,
    "json": SqlType.JSONB,
}

_COLUMNS = """
select TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE,
       CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
from information_schema.COLUMNS
where TABLE_SCHEMA = %s
order by TABLE_NAME, ORDINAL_POSITION
"""

_KEYS = """
select TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME,
       REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
from information_schema.KEY_COLUMN_USAGE
where TABLE_SCHEMA = %s
order by TABLE_NAME, ORDINAL_POSITION
"""


def _map_mysql_type(
    data_type: str,
    column_type: str,
    max_length: int | None,
    precision: int | None,
    scale: int | None,
) -> tuple[SqlType, str | None]:
    """MySQL type to ours.

    tinyint(1) is special-cased because MySQL has no boolean and that is the
    spelling every ORM emits for one. Reading it as an integer would show a
    customer 0 and 1 where they wrote true and false.
    """
    normalised = data_type.strip().lower()

    if normalised == "tinyint" and column_type.replace(" ", "").lower().startswith("tinyint(1)"):
        return SqlType.BOOLEAN, None

    sql_type = _TYPES.get(normalised, SqlType.TEXT)

    if sql_type is SqlType.VARCHAR and max_length:
        return sql_type, str(max_length)
    if sql_type is SqlType.NUMERIC and precision:
        return sql_type, f"{precision},{scale or 0}"
    return sql_type, None


def _build_structure(
    columns: list[dict[str, Any]], keys: list[dict[str, Any]]
) -> DiscoveredStructure:
    by_table: dict[str, list[DiscoveredColumn]] = {}
    for row in columns:
        sql_type, type_arg = _map_mysql_type(
            row["DATA_TYPE"],
            row["COLUMN_TYPE"],
            row["CHARACTER_MAXIMUM_LENGTH"],
            row["NUMERIC_PRECISION"],
            row["NUMERIC_SCALE"],
        )
        by_table.setdefault(row["TABLE_NAME"], []).append(
            DiscoveredColumn(
                name=row["COLUMN_NAME"],
                sql_type=sql_type,
                type_arg=type_arg,
                nullable=row["IS_NULLABLE"] == "YES",
            )
        )

    primary: dict[str, list[str]] = {}
    references: dict[str, list[tuple[str, str, str]]] = {}
    for row in keys:
        table = row["TABLE_NAME"]
        if row["CONSTRAINT_NAME"] == "PRIMARY":
            primary.setdefault(table, []).append(row["COLUMN_NAME"])
        elif row["REFERENCED_TABLE_NAME"]:
            references.setdefault(table, []).append(
                (row["COLUMN_NAME"], row["REFERENCED_TABLE_NAME"], row["REFERENCED_COLUMN_NAME"])
            )

    tables = tuple(
        DiscoveredTable(
            name=name,
            columns=tuple(cols),
            primary_key=tuple(primary[name]) if name in primary else None,
            foreign_keys=tuple(references.get(name, ())),
        )
        for name, cols in sorted(by_table.items())
    )
    return DiscoveredStructure(tables=tables, declared=True)


class MySQLAdapter:
    kind = "mysql"
    supports_incremental = True

    def __init__(self, dsn: str, database: str) -> None:
        self._dsn = dsn
        self._database = database
        self._known: set[str] = set()

    async def _connect(self):
        from urllib.parse import urlparse

        parsed = urlparse(self._dsn)
        return await aiomysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "",
            password=parsed.password or "",
            db=self._database,
        )

    async def discover(self) -> DiscoveredStructure:
        conn = await self._connect()
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(_COLUMNS, (self._database,))
                columns = list(await cursor.fetchall())
                await cursor.execute(_KEYS, (self._database,))
                keys = list(await cursor.fetchall())
        finally:
            conn.close()

        structure = _build_structure(columns, keys)
        self._known = {t.name for t in structure.tables}
        return structure

    async def read(self, table: str, limit: int | None = None) -> Any:
        # Same reasoning as the Postgres adapter: a table name cannot be a
        # bind parameter, so validating against the discovered set is the
        # only barrier against injection.
        if not self._known:
            await self.discover()
        if table not in self._known:
            raise ValueError(f"table {table!r} was not discovered on this source")

        suffix = f" LIMIT {int(limit)}" if limit is not None else ""
        conn = await self._connect()
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(f"SELECT * FROM `{table}`{suffix}")
                rows = list(await cursor.fetchall())
        finally:
            conn.close()
        return pl.DataFrame(rows)
