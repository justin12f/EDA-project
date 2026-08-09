"""Files, through the reader factory that already exists.

`ReaderFactory` already registers .csv, .parquet, .json, .xlsx and .xls for
polars and pandas. This adapter deliberately reimplements none of that — a
new format is a registration there, not a change here.
"""

from __future__ import annotations

import os
from typing import Any

from lumen.architect.adapters.base import DiscoveredColumn, DiscoveredStructure, DiscoveredTable
from lumen.architect.infer import infer_sql_type
from lumen.datasets.materialize import frame_schema
from lumen.readers.exceptions import ReaderError
from lumen.readers.reader_factory import ReaderFactory


class FileAdapter:
    kind = "file"
    # A file is a full snapshot every time; there is no watermark to resume
    # from, so a refresh is always a full reload.
    supports_incremental = False

    def __init__(self, path: str, backend: str = "polars") -> None:
        self._path = path
        self._backend = backend

    @property
    def table_name(self) -> str:
        return os.path.splitext(os.path.basename(self._path))[0]

    def _create_reader(self):
        # ReaderFactory raises ReaderError, which is not a ValueError — this
        # adapter's contract promises ValueError to every caller regardless
        # of source kind, so the translation happens once, here.
        try:
            return ReaderFactory.create(self._path, backend=self._backend)
        except ReaderError as exc:
            raise ValueError(str(exc)) from exc

    async def discover(self) -> DiscoveredStructure:
        frame = self._create_reader().read()
        schema = frame_schema(frame, self._backend)

        columns = []
        for name, dtype in schema.items():
            sql_type, type_arg = infer_sql_type(dtype)
            columns.append(DiscoveredColumn(name=name, sql_type=sql_type, type_arg=type_arg))

        # No primary key and no foreign keys: a file asserts neither. The
        # Architect infers both later, and `declared=False` is what keeps
        # the diagram honest about which is which.
        return DiscoveredStructure(
            tables=(DiscoveredTable(name=self.table_name, columns=tuple(columns)),),
            declared=False,
        )

    async def read(self, table: str, limit: int | None = None) -> Any:
        frame = self._create_reader().read()
        if limit is None:
            return frame
        return frame.head(limit)
