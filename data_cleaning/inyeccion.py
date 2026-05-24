"""Data cleaning dependency injection."""

from __future__ import annotations

from typing import Any

from core.backend import DEFAULT_BACKEND
from core.inyeccion import BackendInyeccionDependency
from data_cleaning.step_factory import (
    AbstractDataCleaningStepFactory,
    PandasDataCleaningStepFactory,
    PolarsDataCleaningStepFactory,
    SparkDataCleaningStepFactory,
)


class DataCleaningInyeccionDependency(BackendInyeccionDependency):
    """Resolves cleaning steps for the bound backend."""

    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    def factory(self) -> type[AbstractDataCleaningStepFactory]:
        return AbstractDataCleaningStepFactory

    def create(self, step_name: str, data_frame: Any, **kwargs: Any) -> Any:
        return AbstractDataCleaningStepFactory.create(
            step_name, data_frame, backend=self._backend, **kwargs
        )

    def create_scoped(
        self,
        step_name: str,
        data_frame: Any,
        columns: list[str],
        **kwargs: Any,
    ) -> Any:
        return AbstractDataCleaningStepFactory.create_scoped(
            step_name, data_frame, columns, backend=self._backend, **kwargs
        )
