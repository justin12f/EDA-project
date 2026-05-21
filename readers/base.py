"""Abstract reader contract — backend-agnostic.

This module defines the contract that ALL reader implementations must
fulfil regardless of the underlying DataFrame library (Polars, PySpark,
or any future backend).

Zero imports from Polars, PySpark, or Pandas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# T represents the DataFrame type of the chosen backend.
# Polars → pl.LazyFrame | pl.DataFrame
# PySpark → pyspark.sql.DataFrame
T = TypeVar("T")


class AbstractReader(ABC, Generic[T]):
    """Contract for reading tabular data from a file into a backend DataFrame.

    Implementations bind ``T`` to their concrete DataFrame type
    (e.g., ``pl.LazyFrame`` for Polars, ``pyspark.sql.DataFrame`` for Spark).

    Args:
        file: Absolute or relative path to the source file.

    Raises:
        readers.exceptions.ReaderError: If the file cannot be read.
    """

    def __init__(self, file: str) -> None:
        self._file = file

    @property
    def file_path(self) -> str:
        """Return the file path this reader was initialised with."""
        return self._file

    @abstractmethod
    def read(self) -> T:
        """Read the source file and return a backend-specific DataFrame.

        Returns:
            A DataFrame in the concrete backend's type.

        Raises:
            ReaderError: On I/O or parsing failures.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(file={self._file!r})"
