"""Module for data analysis"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Verificar si es capa abstracta/legacy; registrar solo contratos en factory maestra y delegar implementaciones a `analyze_data/analyzers/backends/`.
# - ABSTRACCIÓN DEL DATO: Contratos sin tipos pandas fijos; usar TypeVar del contenedor por backend.
# - REFACTOR NATIVO: Eliminar archivo si está obsoleto y sin referencias; si se conserva, solo ABC + registro sin lógica NumPy/Pandas.
# #[AI_CONTEXT_END]
from abc import ABC, abstractmethod

import pandas as pd

class BaseDataAnalysis(ABC):
    """base class for data analysis"""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        self._data_frame = data_frame

    @abstractmethod
    def analyze(self, **kwargs) -> any:
        """analyze the data frame and return the results"""
