"""Module for creation of abstract classes and file reading."""

import os
from abc import ABC, abstractmethod
import pandas as pd


class BaseReader(ABC):
    """BaseClass for all readers"""

    def __init__(self, file: str) -> None:
        self._file = file

    @abstractmethod
    def read(self) -> pd.DataFrame | list[pd.DataFrame]:
        """Read the file"""


class CSVReader(BaseReader):
    """
    Read CSV files with automatic separator detection.

    Uses Python's built-in csv Sniffer (via `sep=None, engine='python'`)
    so it works with comma, semicolon, tab, pipe, and spaced-comma separators
    like those found in dirty_data.csv (`col1 , col2`).
    """

    def read(self) -> pd.DataFrame:
        try:
            # First attempt: auto-detect separator (handles most cases)
            df = pd.read_csv(
                self._file,
                sep=None,
                engine="python",
                skipinitialspace=True,
                na_values=["", "NA", "N/A", "n/a", "null", "NULL", "None", "NaN"],
                keep_default_na=True,
            )
            return df
        except (
            pd.errors.ParserError,
            ValueError,
        ):  # Fallback: standard comma separator
            return pd.read_csv(self._file, skipinitialspace=True)


class ParquetReader(BaseReader):
    """Read Parquet file"""

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self._file)


class JSONReader(BaseReader):
    """Read JSON file — supports both records and split orientation."""

    def read(self) -> pd.DataFrame:
        try:
            return pd.read_json(self._file)
        except ValueError:
            return pd.read_json(self._file, orient="records")


class ExcelReader(BaseReader):
    """Read Excel file (.xlsx / .xls)"""

    def read(self) -> pd.DataFrame:
        return pd.read_excel(self._file)


class HTMLReader(BaseReader):
    """Read the first table found in an HTML file."""

    def read(self) -> pd.DataFrame:
        tables = pd.read_html(self._file)
        if not tables:
            raise ValueError(f"No tables found in {self._file}")
        return tables[0]


class ReaderFactory:
    """Factory for creating readers.

    Usage:
        1. Register a reader: ReaderFactory.register(".ext", ReaderClass)
        2. Create a reader:   ReaderFactory.create("file.ext")
        3. Read the file:     ReaderFactory.create("file.ext").read()
    """

    _registry: dict[str, type[BaseReader]] = {}

    @classmethod
    def register(cls, extension: str, reader: type[BaseReader]) -> None:
        """Register an extension → reader mapping."""
        cls._registry[extension] = reader

    @classmethod
    def create(cls, file: str) -> BaseReader:
        """Return the appropriate reader for the given file extension."""
        extension = os.path.splitext(file)[1].lower()
        reader_class = cls._registry.get(extension)
        if not reader_class:
            raise ValueError(
                f"No reader found for extension '{extension}'. "
                f"Registered: {list(cls._registry.keys())}"
            )
        return reader_class(file)


# Register all built-in readers
ReaderFactory.register(".csv", CSVReader)
ReaderFactory.register(".parquet", ParquetReader)
ReaderFactory.register(".json", JSONReader)
ReaderFactory.register(".xlsx", ExcelReader)
ReaderFactory.register(".xls", ExcelReader)
ReaderFactory.register(".html", HTMLReader)
ReaderFactory.register(".htm", HTMLReader)
