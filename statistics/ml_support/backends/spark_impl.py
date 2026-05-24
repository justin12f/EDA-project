"""Spark statistics backends — `ml_support`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.ml_support.abstract import *

from statistics.ml_support.backends import pandas_impl
from statistics.ml_support.backends.pandas_impl import *

class GiniImpurityCalculatorSpark(AbstractGiniImpurityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GiniImpurityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ImbalanceRatioCalculatorSpark(AbstractImbalanceRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ImbalanceRatioCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class StrategyAdvisorSpark(AbstractStrategyAdvisor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = StrategyAdvisorPandas()

    def advise(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.advise(data, column, **kwargs)

class ClassImbalanceCalculatorSpark(AbstractClassImbalanceCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ClassImbalanceCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CVStrategySelectorSpark(AbstractCVStrategySelector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CVStrategySelectorPandas()

    def select(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class CVConfidenceIntervalCalculatorSpark(AbstractCVConfidenceIntervalCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CVConfidenceIntervalCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CrossValidationCalculatorSpark(AbstractCrossValidationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CrossValidationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CovarianceMatrixComputerSpark(AbstractCovarianceMatrixComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CovarianceMatrixComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class EigenDecompositionCalculatorSpark(AbstractEigenDecompositionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EigenDecompositionCalculatorPandas()

    def decompose(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.decompose(data, column, **kwargs)

class OptimalComponentSelectorSpark(AbstractOptimalComponentSelector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = OptimalComponentSelectorPandas()

    def select(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class DimensionalityReductionCalculatorSpark(AbstractDimensionalityReductionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DimensionalityReductionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseImportanceExtractorSpark(AbstractBaseImportanceExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseImportanceExtractorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class GiniImportanceExtractorSpark(AbstractGiniImportanceExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GiniImportanceExtractorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class PermutationImportanceExtractorSpark(AbstractPermutationImportanceExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PermutationImportanceExtractorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class ImportanceRankerSpark(AbstractImportanceRanker[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ImportanceRankerPandas()

    def rank(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.rank(data, column, **kwargs)

class FeatureImportanceCalculatorSpark(AbstractFeatureImportanceCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FeatureImportanceCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseFeatureScoringMethodSpark(AbstractBaseFeatureScoringMethod[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseFeatureScoringMethodPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class Chi2ScorerSpark(AbstractChi2Scorer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = Chi2ScorerPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class ANOVAFScorerSpark(AbstractANOVAFScorer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ANOVAFScorerPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class MIScorerSpark(AbstractMIScorer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MIScorerPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def score(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class ScoreRankerSpark(AbstractScoreRanker[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ScoreRankerPandas()

    def rank(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.rank(data, column, **kwargs)

class FeatureSelectionCalculatorSpark(AbstractFeatureSelectionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FeatureSelectionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VarianceComputerSpark(AbstractVarianceComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = VarianceComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class FrequencyRatioComputerSpark(AbstractFrequencyRatioComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FrequencyRatioComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class NearZeroVarianceClassifierSpark(AbstractNearZeroVarianceClassifier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NearZeroVarianceClassifierPandas()

    def classify(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.classify(data, column, **kwargs)

class FeatureVarianceCalculatorSpark(AbstractFeatureVarianceCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FeatureVarianceCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BiasVarianceDiagnosticSpark(AbstractBiasVarianceDiagnostic[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BiasVarianceDiagnosticPandas()

    def classify(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.classify(data, column, **kwargs)

class TrainingSizeGeneratorSpark(AbstractTrainingSizeGenerator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TrainingSizeGeneratorPandas()

    def generate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.generate(data, column, **kwargs)

class LearningCurveCalculatorSpark(AbstractLearningCurveCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LearningCurveCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ResidualNormalityCheckerSpark(AbstractResidualNormalityChecker[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ResidualNormalityCheckerPandas()

    def check(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.check(data, column, **kwargs)

class HomoscedasticityCheckerSpark(AbstractHomoscedasticityChecker[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HomoscedasticityCheckerPandas()

    def check(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.check(data, column, **kwargs)

class AutocorrelationCheckerSpark(AbstractAutocorrelationChecker[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AutocorrelationCheckerPandas()

    def check(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.check(data, column, **kwargs)

class ModelResidualsCalculatorSpark(AbstractModelResidualsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ModelResidualsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
