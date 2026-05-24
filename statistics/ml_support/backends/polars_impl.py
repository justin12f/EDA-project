"""Polars statistics backends — `ml_support`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.ml_support.backends import pandas_impl
from statistics.ml_support.backends.pandas_impl import *

from statistics.ml_support.abstract import *

class GiniImpurityCalculatorPolars(AbstractGiniImpurityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GiniImpurityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ImbalanceRatioCalculatorPolars(AbstractImbalanceRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ImbalanceRatioCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class StrategyAdvisorPolars(AbstractStrategyAdvisor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = StrategyAdvisorPandas()

    def advise(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.advise(data, column, **kwargs)

class ClassImbalanceCalculatorPolars(AbstractClassImbalanceCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ClassImbalanceCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CVStrategySelectorPolars(AbstractCVStrategySelector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CVStrategySelectorPandas()

    def select(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class CVConfidenceIntervalCalculatorPolars(AbstractCVConfidenceIntervalCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CVConfidenceIntervalCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CrossValidationCalculatorPolars(AbstractCrossValidationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CrossValidationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CovarianceMatrixComputerPolars(AbstractCovarianceMatrixComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CovarianceMatrixComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class EigenDecompositionCalculatorPolars(AbstractEigenDecompositionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EigenDecompositionCalculatorPandas()

    def decompose(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.decompose(data, column, **kwargs)

class OptimalComponentSelectorPolars(AbstractOptimalComponentSelector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = OptimalComponentSelectorPandas()

    def select(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class DimensionalityReductionCalculatorPolars(AbstractDimensionalityReductionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DimensionalityReductionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseImportanceExtractorPolars(AbstractBaseImportanceExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseImportanceExtractorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class GiniImportanceExtractorPolars(AbstractGiniImportanceExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GiniImportanceExtractorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class PermutationImportanceExtractorPolars(AbstractPermutationImportanceExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PermutationImportanceExtractorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class ImportanceRankerPolars(AbstractImportanceRanker[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ImportanceRankerPandas()

    def rank(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.rank(data, column, **kwargs)

class FeatureImportanceCalculatorPolars(AbstractFeatureImportanceCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FeatureImportanceCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseFeatureScoringMethodPolars(AbstractBaseFeatureScoringMethod[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseFeatureScoringMethodPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class Chi2ScorerPolars(AbstractChi2Scorer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = Chi2ScorerPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class ANOVAFScorerPolars(AbstractANOVAFScorer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ANOVAFScorerPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class MIScorerPolars(AbstractMIScorer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MIScorerPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class ScoreRankerPolars(AbstractScoreRanker[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ScoreRankerPandas()

    def rank(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.rank(data, column, **kwargs)

class FeatureSelectionCalculatorPolars(AbstractFeatureSelectionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FeatureSelectionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VarianceComputerPolars(AbstractVarianceComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = VarianceComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class FrequencyRatioComputerPolars(AbstractFrequencyRatioComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FrequencyRatioComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class NearZeroVarianceClassifierPolars(AbstractNearZeroVarianceClassifier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NearZeroVarianceClassifierPandas()

    def classify(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        frame = eager(data)
        s = numeric_series(frame, column)
        if s.len() < 8:
            raise ValueError('Need at least 8 samples for classify')
        skew = float(s.skew())
        kurt = float(s.kurtosis())
        label = 'symmetric' if abs(skew) < 0.5 else 'skewed'
        return {
            "classification_label": label,
            "skewness": skew,
            "kurtosis": kurt,
            "is_bimodal": False,
            "recommended_transformation": "log1p" if skew > 1 else "none",
        }

class FeatureVarianceCalculatorPolars(AbstractFeatureVarianceCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FeatureVarianceCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BiasVarianceDiagnosticPolars(AbstractBiasVarianceDiagnostic[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BiasVarianceDiagnosticPandas()

    def classify(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        frame = eager(data)
        s = numeric_series(frame, column)
        if s.len() < 8:
            raise ValueError('Need at least 8 samples for classify')
        skew = float(s.skew())
        kurt = float(s.kurtosis())
        label = 'symmetric' if abs(skew) < 0.5 else 'skewed'
        return {
            "classification_label": label,
            "skewness": skew,
            "kurtosis": kurt,
            "is_bimodal": False,
            "recommended_transformation": "log1p" if skew > 1 else "none",
        }

class TrainingSizeGeneratorPolars(AbstractTrainingSizeGenerator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TrainingSizeGeneratorPandas()

    def generate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.generate(data, column, **kwargs)

class LearningCurveCalculatorPolars(AbstractLearningCurveCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LearningCurveCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ResidualNormalityCheckerPolars(AbstractResidualNormalityChecker[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ResidualNormalityCheckerPandas()

    def check(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.check(data, column, **kwargs)

class HomoscedasticityCheckerPolars(AbstractHomoscedasticityChecker[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HomoscedasticityCheckerPandas()

    def check(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.check(data, column, **kwargs)

class AutocorrelationCheckerPolars(AbstractAutocorrelationChecker[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AutocorrelationCheckerPandas()

    def check(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.check(data, column, **kwargs)

class ModelResidualsCalculatorPolars(AbstractModelResidualsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ModelResidualsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
