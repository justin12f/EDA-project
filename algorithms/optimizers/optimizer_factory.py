"""Optimizer factory by backend."""

from __future__ import annotations

from typing import Any

from core.abstract_factory import RegistryFactory
from algorithms.optimizers.gradient_descent import GradientDescent


class _PandasGradientDescent:
    """Wraps legacy GradientDescent for pandas/numpy paths."""

    def __init__(self, **kwargs: Any) -> None:
        self._inner = GradientDescent(**kwargs)

    def optimize(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.optimize(*args, **kwargs)


class OptimizerFactory(RegistryFactory[str, Any]):
    """Registry (optimizer_name, backend) → optimizer."""


def _register() -> None:
    for backend in ("pandas", "polars", "spark"):
        OptimizerFactory.register("gradient_descent", backend, _PandasGradientDescent)


_register()
