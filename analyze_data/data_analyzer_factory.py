"""Module for data analysis factory"""

import pandas as pd

from .analyzers.base import BaseDataAnalysis
from .analyzers.implementations import (
    AnalyseDataColumns,
    AnalyseDataDescribe,
    AnalyseDataHead,
    AnalyseDataIndex,
    AnalyseDataInfo,
    AnalyseDataSample,
    AnalyseDataShape,
    AnalyseDataTail,
    AnalyseDataTypes,
    AnalyseSeasonality,
    AnalyseTrendPatterns,
)


class AnalyzerFactory:
    """Factory for creating data analyzer


    Usage:
    1. Register an analyzer: AnalizerFactory.register("name", AnalyzerClass)
    2. Create an analyzer:   AnalizerFactory.create("name", data_frame)
    3. Analyze the data:     AnalizerFactory.create("name", data_frame).analyze()
    """

    _registry: dict[str, type[BaseDataAnalysis]] = {}

    @classmethod
    def register(cls, name: str, analyzer: type[BaseDataAnalysis]) -> None:
        """Register a new analyzer"""
        cls._registry[name] = analyzer

    @classmethod
    def create(cls, name: str, data_frame: pd.DataFrame) -> BaseDataAnalysis:
        """Create a new analyzer"""
        analyzer = cls._registry.get(name)
        if not analyzer:
            raise ValueError(f"Analyzer {name} not registered")
        return analyzer(data_frame)


AnalyzerFactory.register("types", AnalyseDataTypes)
AnalyzerFactory.register("shape", AnalyseDataShape)
AnalyzerFactory.register("info", AnalyseDataInfo)
AnalyzerFactory.register("describe", AnalyseDataDescribe)
AnalyzerFactory.register("columns", AnalyseDataColumns)
AnalyzerFactory.register("index", AnalyseDataIndex)
AnalyzerFactory.register("head", AnalyseDataHead)
AnalyzerFactory.register("tail", AnalyseDataTail)
AnalyzerFactory.register("sample", AnalyseDataSample)
AnalyzerFactory.register("trend_patterns", AnalyseTrendPatterns)
AnalyzerFactory.register("seasonality", AnalyseSeasonality)
