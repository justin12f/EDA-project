"""Bridge analyzers to statistics triple-backend registry."""

from __future__ import annotations

from typing import Any

from lumen.statistics.inyeccion import StatisticsInyeccionDependency


def run_statistics(
    domain: str,
    calculator: str,
    data: Any,
    *,
    backend: str = "polars",
    column: str | None = None,
    method: str = "calculate",
    **kwargs: Any,
) -> Any:
    """Run a registered statistics calculator on the active backend."""
    return StatisticsInyeccionDependency(backend).run(
        domain, calculator, data, column=column, method=method, **kwargs
    )
