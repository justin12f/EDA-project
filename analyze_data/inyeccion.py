"""Analyze-data dependency injection."""

from __future__ import annotations

from typing import Any

from core.backend import DEFAULT_BACKEND
from core.inyeccion import BackendInyeccionDependency
from analyze_data.analyzer_factory import DataAnalyzerFactory


class AnalyzeDataInyeccionDependency(BackendInyeccionDependency):
    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    def create(self, analyzer_name: str, data_frame: Any = None) -> Any:
        return DataAnalyzerFactory.create_analyzer(
            analyzer_name, self._backend, data_frame
        )
