"""Shared contracts for triple-backend architecture."""

from core.backend import Backend, DEFAULT_BACKEND, validate_backend
from core.abstract_factory import RegistryFactory

__all__ = ["Backend", "DEFAULT_BACKEND", "validate_backend", "RegistryFactory"]
