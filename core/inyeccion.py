"""Base class for backend-scoped dependency injection wrappers."""

from __future__ import annotations

from abc import ABC

from core.backend import Backend, DEFAULT_BACKEND, validate_backend


class BackendInyeccionDependency(ABC):
    """Holds the active backend for a dependency subtree."""

    def __init__(self, backend: Backend | str = DEFAULT_BACKEND) -> None:
        self._backend: Backend = validate_backend(str(backend))

    @property
    def backend(self) -> Backend:
        return self._backend
