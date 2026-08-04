"""Abstract encoder contract — backend-agnostic.

Zero imports from Polars, PySpark, or Pandas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# T represents the DataFrame type of the chosen backend.
T = TypeVar("T")


class AbstractEncoder(ABC, Generic[T]):
    """Contract for categorical encoding transformations.

    Lifecycle:
        1. ``fit(data, **kwargs)`` — learn categories / mappings.
        2. ``transform()`` — return the encoded DataFrame.

    Implementations bind ``T`` to their concrete DataFrame type.
    """

    @abstractmethod
    def fit(self, data: T, **kwargs: Any) -> None:
        """Learn encoding mappings from the data.

        Args:
            data: Backend-specific DataFrame.
            **kwargs: Encoder-specific options (e.g., category hierarchy).

        Raises:
            ValueError: If data is empty or incompatible.
        """

    @abstractmethod
    def transform(self) -> T:
        """Apply the learned encoding.

        Returns:
            Encoded DataFrame in the same backend type.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the encoder has been fitted."""


class AbstractTransform(ABC, Generic[T]):
    """Contract for stateless transform logic.

    Used internally by encoders to separate transform mechanics
    from the fit-state lifecycle.
    """

    @abstractmethod
    def transform(self) -> T:
        """Execute the transformation.

        Returns:
            Transformed DataFrame in the same backend type.
        """
