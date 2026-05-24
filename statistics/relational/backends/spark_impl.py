"""Spark statistics backends — `relational`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.relational.abstract import *

from statistics.relational.backends import pandas_impl
from statistics.relational.backends.pandas_impl import *

class OddsRatioCalculatorSpark(AbstractOddsRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = OddsRatioCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RelativeRiskCalculatorSpark(AbstractRelativeRiskCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RelativeRiskCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ContingencyAnalysisCalculatorSpark(AbstractContingencyAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ContingencyAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CorrelationMatrixComputerSpark(AbstractCorrelationMatrixComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationMatrixComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class CorrelationPairExtractorSpark(AbstractCorrelationPairExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationPairExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class HighCorrelationFlagDetectorSpark(AbstractHighCorrelationFlagDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HighCorrelationFlagDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class CorrelationMatrixCalculatorSpark(AbstractCorrelationMatrixCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationMatrixCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SeriesStandardizerSpark(AbstractSeriesStandardizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SeriesStandardizerPandas()

    def standardize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.standardize(data, column, **kwargs)

class LaggedCorrelationComputerSpark(AbstractLaggedCorrelationComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LaggedCorrelationComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class PeakLagDetectorSpark(AbstractPeakLagDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PeakLagDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class CrossCorrelationCalculatorSpark(AbstractCrossCorrelationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CrossCorrelationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class LaggedDesignMatrixBuilderSpark(AbstractLaggedDesignMatrixBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LaggedDesignMatrixBuilderPandas()

    def build_restricted(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build_restricted(data, column, **kwargs)

    def build_unrestricted(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build_unrestricted(data, column, **kwargs)

class OLSResidualCalculatorSpark(AbstractOLSResidualCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = OLSResidualCalculatorPandas()

    def residual_ss(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.residual_ss(data, column, **kwargs)

class GrangerFStatisticCalculatorSpark(AbstractGrangerFStatisticCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GrangerFStatisticCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GrangerCausalityCalculatorSpark(AbstractGrangerCausalityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GrangerCausalityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AdditiveModelEvaluatorSpark(AbstractAdditiveModelEvaluator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AdditiveModelEvaluatorPandas()

    def evaluate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.evaluate(data, column, **kwargs)

class InteractionModelEvaluatorSpark(AbstractInteractionModelEvaluator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = InteractionModelEvaluatorPandas()

    def evaluate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.evaluate(data, column, **kwargs)

class InteractionGainClassifierSpark(AbstractInteractionGainClassifier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = InteractionGainClassifierPandas()

    def is_meaningful(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.is_meaningful(data, column, **kwargs)

class InteractionEffectsCalculatorSpark(AbstractInteractionEffectsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = InteractionEffectsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VIFRiskClassifierSpark(AbstractVIFRiskClassifier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = VIFRiskClassifierPandas()

    def classify(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.classify(data, column, **kwargs)

class SingleFeatureVIFCalculatorSpark(AbstractSingleFeatureVIFCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SingleFeatureVIFCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MulticollinearityCalculatorSpark(AbstractMulticollinearityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MulticollinearityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TargetTypeDetectorSpark(AbstractTargetTypeDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TargetTypeDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class MIScoreNormalizerSpark(AbstractMIScoreNormalizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MIScoreNormalizerPandas()

    def normalize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class MutualInformationCalculatorSpark(AbstractMutualInformationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MutualInformationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ResidualExtractorSpark(AbstractResidualExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ResidualExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class PartialCorrelationCalculatorSpark(AbstractPartialCorrelationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PartialCorrelationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
