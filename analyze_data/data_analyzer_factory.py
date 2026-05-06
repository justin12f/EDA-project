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
    AnalyseDistributionType,
    AnalyseSkewnessKurtosis,
    AnalyseNormalityTests,
    AnalyseValueCounts,
    AnalysePercentiles,
    AnalyseFrequencyDistribution,
    AnalyseCentralTendency,
    AnalyseDispersion,
    AnalyseHypothesisTest,
    AnalyseAnova,
    AnalyseChiSquare,
    AnalyseCorrelationSignificance,
    AnalyseConfidenceIntervals,
    AnalyseEffectSize,
    AnalysePowerAnalysis,
    AnalyseBootstrap,
    AnalyseCorrelationMatrix,
    AnalyseMulticollinearity,
    AnalyseMutualInformation,
    AnalysePartialCorrelation,
    AnalyseCrossCorrelation,
    AnalyseGrangerCausality,
    AnalyseContingency,
    AnalyseInteractionEffects,
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

# Domain 1 — Descriptive Statistics
AnalyzerFactory.register("distribution_type", AnalyseDistributionType)
AnalyzerFactory.register("skewness_kurtosis", AnalyseSkewnessKurtosis)
AnalyzerFactory.register("normality_tests", AnalyseNormalityTests)
AnalyzerFactory.register("value_counts", AnalyseValueCounts)
AnalyzerFactory.register("percentiles", AnalysePercentiles)
AnalyzerFactory.register("frequency_distribution", AnalyseFrequencyDistribution)
AnalyzerFactory.register("central_tendency", AnalyseCentralTendency)
AnalyzerFactory.register("dispersion", AnalyseDispersion)

# Domain 2 — Inferential Statistics
AnalyzerFactory.register("hypothesis_test", AnalyseHypothesisTest)
AnalyzerFactory.register("anova", AnalyseAnova)
AnalyzerFactory.register("chi_square", AnalyseChiSquare)
AnalyzerFactory.register("correlation_significance", AnalyseCorrelationSignificance)
AnalyzerFactory.register("confidence_intervals", AnalyseConfidenceIntervals)
AnalyzerFactory.register("effect_size", AnalyseEffectSize)
AnalyzerFactory.register("power_analysis", AnalysePowerAnalysis)
AnalyzerFactory.register("bootstrap", AnalyseBootstrap)

# Domain 3 — Relational Statistics
AnalyzerFactory.register("correlation_matrix",      AnalyseCorrelationMatrix)
AnalyzerFactory.register("multicollinearity",        AnalyseMulticollinearity)
AnalyzerFactory.register("mutual_information",       AnalyseMutualInformation)
AnalyzerFactory.register("partial_correlation",      AnalysePartialCorrelation)
AnalyzerFactory.register("cross_correlation",        AnalyseCrossCorrelation)
AnalyzerFactory.register("granger_causality",        AnalyseGrangerCausality)
AnalyzerFactory.register("contingency_analysis",     AnalyseContingency)
AnalyzerFactory.register("interaction_effects",      AnalyseInteractionEffects)
