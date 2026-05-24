"""Backend identifiers and validation."""

from __future__ import annotations

from typing import Literal, TypeAlias

Backend: TypeAlias = Literal["pandas", "polars", "spark"]
DEFAULT_BACKEND: Backend = "polars"
BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "spark")


def validate_backend(backend: str) -> Backend:
    """Return backend if supported; raise ValueError otherwise."""
    if backend not in BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Choose one of: {list(BACKENDS)}"
        )
    return backend  # type: ignore[return-value]
