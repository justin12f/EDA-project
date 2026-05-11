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
    AnalyseFeatureVariance,
    AnalyseFeatureSelection,
    AnalyseFeatureImportance,
    AnalyseDimensionalityReduction,
    AnalyseClassImbalance,
    AnalyseModelResiduals,
    AnalyseLearningCurve,
    AnalyseCrossValidation,
    AnalyseVolatility,
    AnalyseMomentum,
    AnalyseMovingAverages,
    AnalyseStationarity,
    AnalyseLagFeatures,
    AnalyseChangePoints,
    AnalyseForecastAccuracy,
    AnalyseCyclicalPatterns,
    AnalyseRollingStatistics,
    AnalyseKMeansClusters,
    AnalyseRFMSegmentation,
    AnalyseCohortAnalysis,
    AnalysePopulationSplits,
    AnalyseDBSCANClusters,
    AnalyseHierarchicalClusters,
    AnalyseTextBasicStats,
    AnalyseWordFrequency,
    AnalyseSentiment,
    AnalyseTopicDetection,
    AnalyseLanguageDetection,
    AnalyseTextSimilarity,
    AnalyseNamedEntityDensity,
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

# Domain 5 — ML Support
AnalyzerFactory.register("feature_variance",          AnalyseFeatureVariance)
AnalyzerFactory.register("feature_selection",         AnalyseFeatureSelection)
AnalyzerFactory.register("feature_importance",        AnalyseFeatureImportance)
AnalyzerFactory.register("dimensionality_reduction",  AnalyseDimensionalityReduction)
AnalyzerFactory.register("class_imbalance",           AnalyseClassImbalance)
AnalyzerFactory.register("model_residuals",           AnalyseModelResiduals)
AnalyzerFactory.register("learning_curve",            AnalyseLearningCurve)
AnalyzerFactory.register("cross_validation",          AnalyseCrossValidation)

# Domain 4 — Time Series
AnalyzerFactory.register("volatility",          AnalyseVolatility)
AnalyzerFactory.register("momentum",            AnalyseMomentum)
AnalyzerFactory.register("moving_averages",     AnalyseMovingAverages)
AnalyzerFactory.register("stationarity",        AnalyseStationarity)
AnalyzerFactory.register("lag_features",        AnalyseLagFeatures)
AnalyzerFactory.register("change_points",       AnalyseChangePoints)
AnalyzerFactory.register("forecast_accuracy",   AnalyseForecastAccuracy)
AnalyzerFactory.register("cyclical_patterns",   AnalyseCyclicalPatterns)
AnalyzerFactory.register("rolling_statistics",  AnalyseRollingStatistics)

# Domain 7 — Segmentation
AnalyzerFactory.register("kmeans_clusters",        AnalyseKMeansClusters)
AnalyzerFactory.register("rfm_segmentation",       AnalyseRFMSegmentation)
AnalyzerFactory.register("cohort_analysis",        AnalyseCohortAnalysis)
AnalyzerFactory.register("population_splits",      AnalysePopulationSplits)
AnalyzerFactory.register("dbscan_clusters",        AnalyseDBSCANClusters)
AnalyzerFactory.register("hierarchical_clusters",  AnalyseHierarchicalClusters)

# Domain 6 — NLP
AnalyzerFactory.register("text_basic_stats",      AnalyseTextBasicStats)
AnalyzerFactory.register("word_frequency",         AnalyseWordFrequency)
AnalyzerFactory.register("sentiment_analysis",     AnalyseSentiment)
AnalyzerFactory.register("topic_detection",        AnalyseTopicDetection)
AnalyzerFactory.register("language_detection",     AnalyseLanguageDetection)
AnalyzerFactory.register("text_similarity",        AnalyseTextSimilarity)
AnalyzerFactory.register("named_entity_density",   AnalyseNamedEntityDensity)
