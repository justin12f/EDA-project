"""Pandas reader implementations."""

from __future__ import annotations

import pandas as pd

from readers.base import AbstractReader
from readers.exceptions import ReaderError


class PandasCSVReader(AbstractReader[pd.DataFrame]):
    def read(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self._file)
        except Exception as exc:
            raise ReaderError(f"Failed to read CSV: {self._file}") from exc


class PandasParquetReader(AbstractReader[pd.DataFrame]):
    def read(self) -> pd.DataFrame:
        try:
            return pd.read_parquet(self._file)
        except Exception as exc:
            raise ReaderError(f"Failed to read Parquet: {self._file}") from exc


class PandasJSONReader(AbstractReader[pd.DataFrame]):
    def read(self) -> pd.DataFrame:
        try:
            return pd.read_json(self._file)
        except Exception as exc:
            raise ReaderError(f"Failed to read JSON: {self._file}") from exc


class PandasExcelReader(AbstractReader[pd.DataFrame]):
    def read(self) -> pd.DataFrame:
        try:
            return pd.read_excel(self._file)
        except Exception as exc:
            raise ReaderError(f"Failed to read Excel: {self._file}") from exc
