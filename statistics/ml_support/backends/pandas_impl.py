"""Pandas statistics backends — `ml_support`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.ml_support.abstract import *

import statistics.ml_support.class_imbalance as _mod_class_imbalance
import statistics.ml_support.cross_validation as _mod_cross_validation
import statistics.ml_support.dimensionality_reduction as _mod_dimensionality_reduction
import statistics.ml_support.feature_importance as _mod_feature_importance
import statistics.ml_support.feature_selection as _mod_feature_selection
import statistics.ml_support.feature_variance as _mod_feature_variance
import statistics.ml_support.learning_curve as _mod_learning_curve
import statistics.ml_support.model_residuals as _mod_model_residuals

class GiniImpurityCalculatorPandas(AbstractGiniImpurityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_class_imbalance.GiniImpurityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ImbalanceRatioCalculatorPandas(AbstractImbalanceRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_class_imbalance.ImbalanceRatioCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class StrategyAdvisorPandas(AbstractStrategyAdvisor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_class_imbalance.StrategyAdvisor()

    def advise(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.advise(arr, **kwargs)

class ClassImbalanceCalculatorPandas(AbstractClassImbalanceCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_class_imbalance.ClassImbalanceCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CVStrategySelectorPandas(AbstractCVStrategySelector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cross_validation.CVStrategySelector()

    def select(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.select(arr, **kwargs)

class CVConfidenceIntervalCalculatorPandas(AbstractCVConfidenceIntervalCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cross_validation.CVConfidenceIntervalCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CrossValidationCalculatorPandas(AbstractCrossValidationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_cross_validation.CrossValidationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CovarianceMatrixComputerPandas(AbstractCovarianceMatrixComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dimensionality_reduction.CovarianceMatrixComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class EigenDecompositionCalculatorPandas(AbstractEigenDecompositionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dimensionality_reduction.EigenDecompositionCalculator()

    def decompose(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.decompose(arr, **kwargs)

class OptimalComponentSelectorPandas(AbstractOptimalComponentSelector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dimensionality_reduction.OptimalComponentSelector()

    def select(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.select(arr, **kwargs)

class DimensionalityReductionCalculatorPandas(AbstractDimensionalityReductionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dimensionality_reduction.DimensionalityReductionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BaseImportanceExtractorPandas(AbstractBaseImportanceExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_importance.BaseImportanceExtractor()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class GiniImportanceExtractorPandas(AbstractGiniImportanceExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_importance.GiniImportanceExtractor()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class PermutationImportanceExtractorPandas(AbstractPermutationImportanceExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_importance.PermutationImportanceExtractor()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class ImportanceRankerPandas(AbstractImportanceRanker[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_importance.ImportanceRanker()

    def rank(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.rank(arr, **kwargs)

class FeatureImportanceCalculatorPandas(AbstractFeatureImportanceCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_importance.FeatureImportanceCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BaseFeatureScoringMethodPandas(AbstractBaseFeatureScoringMethod[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_selection.BaseFeatureScoringMethod()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def score(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.score(arr, **kwargs)

class Chi2ScorerPandas(AbstractChi2Scorer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_selection.Chi2Scorer()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def score(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.score(arr, **kwargs)

class ANOVAFScorerPandas(AbstractANOVAFScorer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_selection.ANOVAFScorer()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def score(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.score(arr, **kwargs)

class MIScorerPandas(AbstractMIScorer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_selection.MIScorer()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def score(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.score(arr, **kwargs)

class ScoreRankerPandas(AbstractScoreRanker[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_selection.ScoreRanker()

    def rank(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.rank(arr, **kwargs)

class FeatureSelectionCalculatorPandas(AbstractFeatureSelectionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_selection.FeatureSelectionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class VarianceComputerPandas(AbstractVarianceComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_variance.VarianceComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class FrequencyRatioComputerPandas(AbstractFrequencyRatioComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_variance.FrequencyRatioComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class NearZeroVarianceClassifierPandas(AbstractNearZeroVarianceClassifier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_variance.NearZeroVarianceClassifier()

    def classify(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.classify(arr, **kwargs)

class FeatureVarianceCalculatorPandas(AbstractFeatureVarianceCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_feature_variance.FeatureVarianceCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BiasVarianceDiagnosticPandas(AbstractBiasVarianceDiagnostic[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_learning_curve.BiasVarianceDiagnostic()

    def classify(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.classify(arr, **kwargs)

class TrainingSizeGeneratorPandas(AbstractTrainingSizeGenerator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_learning_curve.TrainingSizeGenerator()

    def generate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.generate(arr, **kwargs)

class LearningCurveCalculatorPandas(AbstractLearningCurveCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_learning_curve.LearningCurveCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ResidualNormalityCheckerPandas(AbstractResidualNormalityChecker[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_model_residuals.ResidualNormalityChecker()

    def check(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.check(arr, **kwargs)

class HomoscedasticityCheckerPandas(AbstractHomoscedasticityChecker[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_model_residuals.HomoscedasticityChecker()

    def check(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.check(arr, **kwargs)

class AutocorrelationCheckerPandas(AbstractAutocorrelationChecker[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_model_residuals.AutocorrelationChecker()

    def check(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.check(arr, **kwargs)

class ModelResidualsCalculatorPandas(AbstractModelResidualsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_model_residuals.ModelResidualsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
