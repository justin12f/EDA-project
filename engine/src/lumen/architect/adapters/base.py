"""What every source must be able to answer.

Two questions: what is your structure, and give me a table. A file infers
its structure; a live database reads its own. `declared` is how the caller
tells those apart, and it is the difference between a diagram showing a
constraint the database actually holds and one showing a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from lumen.architect.spec import SqlType


@dataclass(frozen=True)
class DiscoveredColumn:
    name: str
    sql_type: SqlType
    type_arg: str | None = None
    nullable: bool = True


@dataclass(frozen=True)
class DiscoveredTable:
    name: str
    columns: tuple[DiscoveredColumn, ...]
    primary_key: tuple[str, ...] | None = None
    # (from_column, to_table, to_column) — populated only when the source
    # declares its own constraints.
    foreign_keys: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class DiscoveredStructure:
    tables: tuple[DiscoveredTable, ...]
    declared: bool


@runtime_checkable
class SourceAdapter(Protocol):
    kind: str
    supports_incremental: bool

    async def discover(self) -> DiscoveredStructure: ...

    async def read(self, table: str, limit: int | None = None) -> Any: ...
