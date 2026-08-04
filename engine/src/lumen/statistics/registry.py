"""Unified statistics factory registry."""
from __future__ import annotations
from typing import Any
from lumen.core.abstract_factory import RegistryFactory

class StatisticsRegistry(RegistryFactory[str, Any]):
    """Keys: `{domain}.{calculator}`"""

DOMAIN_FACTORIES: dict[str, type] = {}

def _register_all() -> None:
    from lumen.statistics.business.factory import BusinessStatisticsFactory
    DOMAIN_FACTORIES['business'] = BusinessStatisticsFactory
    for key in BusinessStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if BusinessStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('business.' + key, backend, BusinessStatisticsFactory.get_class(key, backend))
    from lumen.statistics.descriptive.factory import DescriptiveStatisticsFactory
    DOMAIN_FACTORIES['descriptive'] = DescriptiveStatisticsFactory
    for key in DescriptiveStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if DescriptiveStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('descriptive.' + key, backend, DescriptiveStatisticsFactory.get_class(key, backend))
    from lumen.statistics.geospatial.factory import GeospatialStatisticsFactory
    DOMAIN_FACTORIES['geospatial'] = GeospatialStatisticsFactory
    for key in GeospatialStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if GeospatialStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('geospatial.' + key, backend, GeospatialStatisticsFactory.get_class(key, backend))
    from lumen.statistics.graphs.factory import GraphsStatisticsFactory
    DOMAIN_FACTORIES['graphs'] = GraphsStatisticsFactory
    for key in GraphsStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if GraphsStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('graphs.' + key, backend, GraphsStatisticsFactory.get_class(key, backend))
    from lumen.statistics.inferential.factory import InferentialStatisticsFactory
    DOMAIN_FACTORIES['inferential'] = InferentialStatisticsFactory
    for key in InferentialStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if InferentialStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('inferential.' + key, backend, InferentialStatisticsFactory.get_class(key, backend))
    from lumen.statistics.ml_support.factory import MlSupportStatisticsFactory
    DOMAIN_FACTORIES['ml_support'] = MlSupportStatisticsFactory
    for key in MlSupportStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if MlSupportStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('ml_support.' + key, backend, MlSupportStatisticsFactory.get_class(key, backend))
    from lumen.statistics.nlp.factory import NlpStatisticsFactory
    DOMAIN_FACTORIES['nlp'] = NlpStatisticsFactory
    for key in NlpStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if NlpStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('nlp.' + key, backend, NlpStatisticsFactory.get_class(key, backend))
    from lumen.statistics.relational.factory import RelationalStatisticsFactory
    DOMAIN_FACTORIES['relational'] = RelationalStatisticsFactory
    for key in RelationalStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if RelationalStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('relational.' + key, backend, RelationalStatisticsFactory.get_class(key, backend))
    from lumen.statistics.segmentation.factory import SegmentationStatisticsFactory
    DOMAIN_FACTORIES['segmentation'] = SegmentationStatisticsFactory
    for key in SegmentationStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if SegmentationStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('segmentation.' + key, backend, SegmentationStatisticsFactory.get_class(key, backend))
    from lumen.statistics.survival.factory import SurvivalStatisticsFactory
    DOMAIN_FACTORIES['survival'] = SurvivalStatisticsFactory
    for key in SurvivalStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if SurvivalStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('survival.' + key, backend, SurvivalStatisticsFactory.get_class(key, backend))
    from lumen.statistics.time_series.factory import TimeSeriesStatisticsFactory
    DOMAIN_FACTORIES['time_series'] = TimeSeriesStatisticsFactory
    for key in TimeSeriesStatisticsFactory.registered_keys():
        for backend in ('pandas', 'polars', 'spark'):
            if TimeSeriesStatisticsFactory.is_registered(key, backend):
                StatisticsRegistry.register('time_series.' + key, backend, TimeSeriesStatisticsFactory.get_class(key, backend))

try:
    _register_all()
except ModuleNotFoundError:
    pass
