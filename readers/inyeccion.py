"""Readers dependency injection layer."""

from __future__ import annotations

from typing import Any

from core.backend import Backend, DEFAULT_BACKEND
from core.inyeccion import BackendInyeccionDependency
from readers.reader_factory import ReaderFactory


class ReadersInyeccionDependency(BackendInyeccionDependency):
    """Read files using the backend bound at construction time."""

    def __init__(self, backend: Backend | str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    def read(self, file: str, backend: Backend | str | None = None) -> Any:
        """Read file and return backend-native frame (LazyFrame, Spark DF, or pandas DF)."""
        active = self._backend if backend is None else backend
        reader = ReaderFactory.create(file, backend=active)  # type: ignore[arg-type]
        return reader.read()
