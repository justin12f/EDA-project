"""Statistics package — triple-backend architecture (pandas, polars, spark)."""

from statistics.inyeccion import StatisticsInyeccionDependency
from statistics.registry import DOMAIN_FACTORIES, StatisticsRegistry

__all__ = [
    "StatisticsInyeccionDependency",
    "StatisticsRegistry",
    "DOMAIN_FACTORIES",
]
