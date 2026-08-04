"""Data analyzer factory: (analyzer_name, backend) → analyzer class."""

from __future__ import annotations

from typing import Any

from lumen.core.abstract_factory import RegistryFactory
from lumen.core.backend import Backend
from lumen.analyze_data.analyzers.backends.abstract_analyzers import AbstractBaseDataAnalysis


class DataAnalyzerFactory(RegistryFactory[str, AbstractBaseDataAnalysis[Any]]):
    """Creates analyzer instances for a given name and backend."""

    @classmethod
    def create_analyzer(
        cls,
        name: str,
        backend: Backend | str,
        data_frame: Any = None,
    ) -> AbstractBaseDataAnalysis[Any]:
        analyzer = cls.create(name, backend)
        if data_frame is not None:
            analyzer.set_data_frame(data_frame)
        return analyzer


def _register_analyzers() -> None:
    from lumen.analyze_data.analyzers.backends import pandas_impl, polars_impl, spark_impl

    mapping = {
        "columns": ("AnalyseDataColumns", "AnalyseDataColumnsPolars", "AnalyseDataColumnsSpark"),
        "describe": ("AnalyseDataDescribe", "AnalyseDataDescribePolars", "AnalyseDataDescribeSpark"),
        "head": ("AnalyseDataHead", "AnalyseDataHeadPolars", "AnalyseDataHeadSpark"),
        "index": ("AnalyseDataIndex", "AnalyseDataIndexPolars", "AnalyseDataIndexSpark"),
        "info": ("AnalyseDataInfo", "AnalyseDataInfoPolars", "AnalyseDataInfoSpark"),
        "sample": ("AnalyseDataSample", "AnalyseDataSamplePolars", "AnalyseDataSampleSpark"),
        "shape": ("AnalyseDataShape", "AnalyseDataShapePolars", "AnalyseDataShapeSpark"),
        "tail": ("AnalyseDataTail", "AnalyseDataTailPolars", "AnalyseDataTailSpark"),
        "types": ("AnalyseDataTypes", "AnalyseDataTypesPolars", "AnalyseDataTypesSpark"),
    }
    for key, (pd_name, pl_name, sp_name) in mapping.items():
        DataAnalyzerFactory.register(key, "pandas", getattr(pandas_impl, pd_name))
        DataAnalyzerFactory.register(key, "polars", getattr(polars_impl, pl_name))
        DataAnalyzerFactory.register(key, "spark", getattr(spark_impl, sp_name))


_register_analyzers()

# Legacy alias
AnalyzerFactory = DataAnalyzerFactory
