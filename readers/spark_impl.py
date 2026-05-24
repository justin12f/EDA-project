"""PySpark reader implementations.

All readers return ``pyspark.sql.DataFrame``.
Spark DataFrames are already lazy (DAG of transformations).
"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `ReaderFactory` local por extensión y backend; `ReadersInyeccionDependency` como capa superior para Agentes.
# - ABSTRACCIÓN DEL DATO: Retorno de `read()` como `pl.LazyFrame` (polars), `pyspark.sql.DataFrame` (spark) o `pd.DataFrame` (pandas); registrar readers pandas faltantes en la factory.
# - REFACTOR NATIVO: Ampliar formatos con APIs nativas (`pl.scan_*`, `spark.read`, `pd.read_*`) sin mezclar backends en una misma clase.
# #[AI_CONTEXT_END]
from __future__ import annotations

import os
import sys

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from readers.base import AbstractReader
from readers.exceptions import ReaderError

def _get_or_create_spark(app_name: str = "EDA-Project") -> SparkSession:
    """Return the active SparkSession or create a new local one.

    Sets PYSPARK_PYTHON to the current interpreter so venv works.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .getOrCreate()
    )

class SparkCSVReader(AbstractReader[SparkDataFrame]):
    """Read CSV files into a PySpark DataFrame.

    Uses ``spark.read.csv`` with header and schema inference enabled.
    """

    def read(self) -> SparkDataFrame:
        """Read CSV with auto-detected separator.

        Returns:
            pyspark.sql.DataFrame backed by the CSV file.

        Raises:
            ReaderError: If the file cannot be parsed.
        """
        try:
            spark = _get_or_create_spark()
            return spark.read.csv(
                self._file,
                header=True,
                inferSchema=True,
                nanValue="NaN",
                nullValue="",
            )
        except Exception as exc:
            raise ReaderError(
                f"Failed to read CSV '{self._file}': {exc}"
            ) from exc

class SparkParquetReader(AbstractReader[SparkDataFrame]):
    """Read Parquet files into a PySpark DataFrame."""

    def read(self) -> SparkDataFrame:
        try:
            spark = _get_or_create_spark()
            return spark.read.parquet(self._file)
        except Exception as exc:
            raise ReaderError(
                f"Failed to read Parquet '{self._file}': {exc}"
            ) from exc

class SparkJSONReader(AbstractReader[SparkDataFrame]):
    """Read JSON files into a PySpark DataFrame."""

    def read(self) -> SparkDataFrame:
        try:
            spark = _get_or_create_spark()
            return spark.read.json(self._file)
        except Exception as exc:
            raise ReaderError(
                f"Failed to read JSON '{self._file}': {exc}"
            ) from exc
