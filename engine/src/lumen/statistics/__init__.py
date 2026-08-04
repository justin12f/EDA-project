"""Statistics package — triple-backend architecture (pandas, polars, spark)."""

from lumen.statistics.inyeccion import StatisticsInyeccionDependency
from lumen.statistics.registry import DOMAIN_FACTORIES, StatisticsRegistry

__all__ = [
    "StatisticsInyeccionDependency",
    "StatisticsRegistry",
    "DOMAIN_FACTORIES",
]
