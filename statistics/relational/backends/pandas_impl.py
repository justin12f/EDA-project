"""Pandas statistics backends — `relational`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.relational.abstract import *

import statistics.relational.contingency_analysis as _mod_contingency_analysis
import statistics.relational.correlation_matrix as _mod_correlation_matrix
import statistics.relational.cross_correlation as _mod_cross_correlation
import statistics.relational.granger_causality as _mod_granger_causality
import statistics.relational.interaction_effects as _mod_interaction_effects
import statistics.relational.multicollinearity as _mod_multicollinearity
import statistics.relational.mutual_information as _mod_mutual_information
import statistics.relational.partial_correlation as _mod_partial_correlation

class OddsRatioCalculatorPandas(AbstractOddsRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_contingency_analysis.OddsRatioCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RelativeRiskCalculatorPandas(AbstractRelativeRiskCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_contingency_analysis.RelativeRiskCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ContingencyAnalysisCalculatorPandas(AbstractContingencyAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_contingency_analysis.ContingencyAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CorrelationMatrixComputerPandas(AbstractCorrelationMatrixComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_correlation_matrix.CorrelationMatrixComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class CorrelationPairExtractorPandas(AbstractCorrelationPairExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_correlation_matrix.CorrelationPairExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class HighCorrelationFlagDetectorPandas(AbstractHighCorrelationFlagDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_correlation_matrix.HighCorrelationFlagDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class CorrelationMatrixCalculatorPandas(AbstractCorrelationMatrixCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_correlation_matrix.CorrelationMatrixCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SeriesStandardizerPandas(AbstractSeriesStandardizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cross_correlation.SeriesStandardizer()

    def standardize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.standardize(arr, **kwargs)

class LaggedCorrelationComputerPandas(AbstractLaggedCorrelationComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cross_correlation.LaggedCorrelationComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class PeakLagDetectorPandas(AbstractPeakLagDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cross_correlation.PeakLagDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class CrossCorrelationCalculatorPandas(AbstractCrossCorrelationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cross_correlation.CrossCorrelationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class LaggedDesignMatrixBuilderPandas(AbstractLaggedDesignMatrixBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_granger_causality.LaggedDesignMatrixBuilder()

    def build_restricted(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build_restricted(arr, **kwargs)

    def build_unrestricted(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build_unrestricted(arr, **kwargs)

class OLSResidualCalculatorPandas(AbstractOLSResidualCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_granger_causality.OLSResidualCalculator()

    def residual_ss(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.residual_ss(arr, **kwargs)

class GrangerFStatisticCalculatorPandas(AbstractGrangerFStatisticCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_granger_causality.GrangerFStatisticCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GrangerCausalityCalculatorPandas(AbstractGrangerCausalityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_granger_causality.GrangerCausalityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class AdditiveModelEvaluatorPandas(AbstractAdditiveModelEvaluator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_interaction_effects.AdditiveModelEvaluator()

    def evaluate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.evaluate(arr, **kwargs)

class InteractionModelEvaluatorPandas(AbstractInteractionModelEvaluator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_interaction_effects.InteractionModelEvaluator()

    def evaluate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.evaluate(arr, **kwargs)

class InteractionGainClassifierPandas(AbstractInteractionGainClassifier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_interaction_effects.InteractionGainClassifier()

    def is_meaningful(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.is_meaningful(arr, **kwargs)

class InteractionEffectsCalculatorPandas(AbstractInteractionEffectsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_interaction_effects.InteractionEffectsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class VIFRiskClassifierPandas(AbstractVIFRiskClassifier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_multicollinearity.VIFRiskClassifier()

    def classify(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.classify(arr, **kwargs)

class SingleFeatureVIFCalculatorPandas(AbstractSingleFeatureVIFCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_multicollinearity.SingleFeatureVIFCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MulticollinearityCalculatorPandas(AbstractMulticollinearityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_multicollinearity.MulticollinearityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TargetTypeDetectorPandas(AbstractTargetTypeDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_mutual_information.TargetTypeDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class MIScoreNormalizerPandas(AbstractMIScoreNormalizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_mutual_information.MIScoreNormalizer()

    def normalize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.normalize(arr, **kwargs)

class MutualInformationCalculatorPandas(AbstractMutualInformationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_mutual_information.MutualInformationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ResidualExtractorPandas(AbstractResidualExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_partial_correlation.ResidualExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class PartialCorrelationCalculatorPandas(AbstractPartialCorrelationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_partial_correlation.PartialCorrelationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
