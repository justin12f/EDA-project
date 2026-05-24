"""Abstract statistics calculator contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class AbstractColumnCalculator(ABC, Generic[T]):
    """Compute a scalar or dict statistic from a column in backend frame T."""

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any:
        """Run statistic on ``column`` of ``data``."""
