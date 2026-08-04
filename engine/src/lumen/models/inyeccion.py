"""Models dependency injection."""

from __future__ import annotations

from typing import Any

from lumen.core.backend import DEFAULT_BACKEND
from lumen.core.inyeccion import BackendInyeccionDependency
from lumen.models.linear_regression import LinearRegressionFactory


class ModelsInyeccionDependency(BackendInyeccionDependency):
    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    def linear_regression(self, type_of_prediction: str, complexity: str) -> Any:
        return LinearRegressionFactory.create(type_of_prediction, complexity)

    def optimizer(self, name: str = "gradient_descent", **kwargs: Any) -> Any:
        from lumen.algorithms.optimizers.optimizer_factory import OptimizerFactory

        return OptimizerFactory.create(name, self._backend, **kwargs)
