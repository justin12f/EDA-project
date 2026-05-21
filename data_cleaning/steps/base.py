"""Abstract base classes for data cleaning pipeline steps — backend-agnostic.

Zero imports from Polars, PySpark, or Pandas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# T represents the DataFrame type of the chosen backend.
T = TypeVar("T")


class AbstractStep(ABC, Generic[T]):
    """Contract for a single data cleaning pipeline step.

    Each step receives a DataFrame, transforms it, and returns
    a new DataFrame of the same backend type.

    Implementations bind ``T`` to their concrete DataFrame type.
    """

    def __init__(self, data_frame: T) -> None:
        self._data_frame = data_frame

    @abstractmethod
    def process(self, data: T) -> T:
        """Process the DataFrame and return the cleaned version.

        Args:
            data: Input DataFrame to clean.

        Returns:
            Cleaned DataFrame of the same backend type.
        """
