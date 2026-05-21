"""Reader factory with multi-backend support.

Usage:
    # Polars backend (default)
    reader = ReaderFactory.create("data.csv")
    lf = reader.read()  # → pl.LazyFrame

    # PySpark backend
    reader = ReaderFactory.create("data.csv", backend="spark")
    sdf = reader.read()  # → pyspark.sql.DataFrame

    # Explicit backend
    reader = ReaderFactory.create("data.parquet", backend="polars")
"""

from __future__ import annotations

import os
from typing import Literal

from readers.base import AbstractReader
from readers.exceptions import BackendNotSupportedError, ReaderError

# ── Polars readers ────────────────────────────────────────────────────────────
from readers.polars_impl import (
    PolarsCSVReader,
    PolarsExcelReader,
    PolarsJSONReader,
    PolarsParquetReader,
)

# ── PySpark readers ───────────────────────────────────────────────────────────
from readers.spark_impl import (
    SparkCSVReader,
    SparkJSONReader,
    SparkParquetReader,
)


# Type alias for supported backends
Backend = Literal["polars", "spark"]


class ReaderFactory:
    """Factory for creating readers with multi-backend support.

    The registry maps ``(extension, backend)`` → ``ReaderClass``.

    Usage:
        1. Register:  ``ReaderFactory.register(".csv", "polars", PolarsCSVReader)``
        2. Create:    ``ReaderFactory.create("file.csv", backend="polars")``
        3. Read:      ``ReaderFactory.create("file.csv").read()``
    """

    _registry: dict[tuple[str, str], type[AbstractReader]] = {}

    @classmethod
    def register(
        cls,
        extension: str,
        backend: str,
        reader: type[AbstractReader],
    ) -> None:
        """Register an (extension, backend) → reader mapping."""
        cls._registry[(extension, backend)] = reader

    @classmethod
    def create(
        cls,
        file: str,
        backend: Backend = "polars",
    ) -> AbstractReader:
        """Return the appropriate reader for the file extension and backend.

        Args:
            file: Path to the file to read.
            backend: Backend engine — ``"polars"`` or ``"spark"``.

        Returns:
            An ``AbstractReader`` bound to the given backend.

        Raises:
            BackendNotSupportedError: If the backend is not registered.
            ReaderError: If no reader exists for the file extension.
        """
        extension = os.path.splitext(file)[1].lower()
        key = (extension, backend)
        reader_class = cls._registry.get(key)

        if reader_class is None:
            # Check if the backend exists at all
            available_backends = sorted({b for (_, b) in cls._registry})
            if backend not in available_backends:
                raise BackendNotSupportedError(backend, available_backends)

            # Backend exists but extension is not supported
            supported = sorted(
                ext for (ext, b) in cls._registry if b == backend
            )
            raise ReaderError(
                f"No reader for extension '{extension}' on backend '{backend}'. "
                f"Supported extensions: {supported}"
            )

        return reader_class(file)

    @classmethod
    def available_backends(cls) -> list[str]:
        """Return list of registered backend names."""
        return sorted({b for (_, b) in cls._registry})

    @classmethod
    def available_extensions(cls, backend: str) -> list[str]:
        """Return list of registered extensions for a given backend."""
        return sorted(ext for (ext, b) in cls._registry if b == backend)


# ═══════════════════════════════════════════════════════════════════════════════
# Register all built-in readers
# ═══════════════════════════════════════════════════════════════════════════════

# ── Polars ────────────────────────────────────────────────────────────────────
ReaderFactory.register(".csv", "polars", PolarsCSVReader)
ReaderFactory.register(".parquet", "polars", PolarsParquetReader)
ReaderFactory.register(".json", "polars", PolarsJSONReader)
ReaderFactory.register(".xlsx", "polars", PolarsExcelReader)
ReaderFactory.register(".xls", "polars", PolarsExcelReader)

# ── PySpark ───────────────────────────────────────────────────────────────────
ReaderFactory.register(".csv", "spark", SparkCSVReader)
ReaderFactory.register(".parquet", "spark", SparkParquetReader)
ReaderFactory.register(".json", "spark", SparkJSONReader)
