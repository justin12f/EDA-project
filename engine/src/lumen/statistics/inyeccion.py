"""Central statistics dependency injection."""

from __future__ import annotations

from typing import Any

from lumen.core.backend import DEFAULT_BACKEND
from lumen.core.inyeccion import BackendInyeccionDependency
from lumen.statistics.registry import DOMAIN_FACTORIES, StatisticsRegistry


class StatisticsInyeccionDependency(BackendInyeccionDependency):
    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    def create(
        self, domain: str, calculator_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        key = f"{domain}.{calculator_name}"
        return StatisticsRegistry.create(key, self._backend, *args, **kwargs)

    def run(
        self,
        domain: str,
        calculator_name: str,
        data: Any,
        column: str | None = None,
        method: str = "calculate",
        **kwargs: Any,
    ) -> Any:
        """Invoke calculator method on backend-native frame."""
        calc = self.create(domain, calculator_name)
        fn = getattr(calc, method, None)
        if fn is None:
            raise AttributeError(
                f"{calculator_name} has no method {method!r} on backend {self._backend!r}"
            )
        if column is not None:
            return fn(data, column, **kwargs)
        return fn(data, **kwargs)

    def domain_factory(self, domain: str) -> type:
        if domain not in DOMAIN_FACTORIES:
            raise ValueError(f"Unknown domain '{domain}'. Available: {sorted(DOMAIN_FACTORIES)}")
        return DOMAIN_FACTORIES[domain]
