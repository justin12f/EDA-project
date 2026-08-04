"""Abstract base classes for data cleaning pipeline steps — backend-agnostic.

Zero imports from Polars, PySpark, or Pandas.
"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `DataCleaningStepFactory` del backend correspondiente en `data_cleaning/steps/backends/`; inyección vía `DataCleaningInyeccionDependency` → Factory Maestra.
# - ABSTRACCIÓN DEL DATO: Canonicalizar en `backends/`; deprecar duplicados en `steps/implementations.py` y `steps/polars_impl.py` raíz tras verificar referencias.
# - REFACTOR NATIVO: Steps en inglés y 100 % API nativa del backend; sin NumPy salvo materialización local explícita.
# #[AI_CONTEXT_END]
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
