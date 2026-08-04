"""Polars reader implementations.

All readers return ``pl.LazyFrame`` as the preferred type.
The caller can materialise with ``.collect()`` when needed.
"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `ReaderFactory` local por extensión y backend; `ReadersInyeccionDependency` como capa superior para Agentes.
# - ABSTRACCIÓN DEL DATO: Retorno de `read()` como `pl.LazyFrame` (polars), `pyspark.sql.DataFrame` (spark) o `pd.DataFrame` (pandas); registrar readers pandas faltantes en la factory.
# - REFACTOR NATIVO: Ampliar formatos con APIs nativas (`pl.scan_*`, `spark.read`, `pd.read_*`) sin mezclar backends en una misma clase.
# #[AI_CONTEXT_END]
from __future__ import annotations

import polars as pl

from lumen.readers.base import AbstractReader
from lumen.readers.exceptions import ReaderError

class PolarsCSVReader(AbstractReader[pl.LazyFrame]):
    """Read CSV files into a Polars LazyFrame.

    Uses ``pl.scan_csv`` (lazy) so no data is loaded until
    the query plan is executed with ``.collect()``.
    """

    def read(self) -> pl.LazyFrame:
        """Read CSV — auto-detects separator via Polars heuristics.

        Returns:
            pl.LazyFrame backed by the CSV file.

        Raises:
            ReaderError: If the file cannot be parsed.
        """
        try:
            return pl.scan_csv(
                self._file,
                infer_schema_length=10_000,
                null_values=["", "NA", "N/A", "n/a", "null", "NULL", "None", "NaN"],
                try_parse_dates=True,
            )
        except Exception as exc:
            raise ReaderError(
                f"Failed to read CSV '{self._file}': {exc}"
            ) from exc

class PolarsParquetReader(AbstractReader[pl.LazyFrame]):
    """Read Parquet files into a Polars LazyFrame.

    Uses ``pl.scan_parquet`` for zero-copy lazy reads.
    """

    def read(self) -> pl.LazyFrame:
        try:
            return pl.scan_parquet(self._file)
        except Exception as exc:
            raise ReaderError(
                f"Failed to read Parquet '{self._file}': {exc}"
            ) from exc

class PolarsJSONReader(AbstractReader[pl.LazyFrame]):
    """Read JSON (newline-delimited) files into a Polars LazyFrame.

    Falls back to ``pl.read_json`` (eager) + ``.lazy()`` because
    Polars does not have a native ``scan_json`` for all JSON formats.
    """

    def read(self) -> pl.LazyFrame:
        try:
            return pl.read_json(self._file).lazy()
        except Exception as exc:
            raise ReaderError(
                f"Failed to read JSON '{self._file}': {exc}"
            ) from exc

class PolarsExcelReader(AbstractReader[pl.LazyFrame]):
    """Read Excel (.xlsx / .xls) files into a Polars LazyFrame.

    Requires ``openpyxl`` or ``xlsx2csv`` as an engine dependency.
    """

    def read(self) -> pl.LazyFrame:
        try:
            return pl.read_excel(self._file).lazy()
        except Exception as exc:
            raise ReaderError(
                f"Failed to read Excel '{self._file}': {exc}"
            ) from exc

