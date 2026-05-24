"""Polars statistics backends — `relational`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.relational.backends import pandas_impl
from statistics.relational.backends.pandas_impl import *

from statistics.relational.abstract import *

class OddsRatioCalculatorPolars(AbstractOddsRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = OddsRatioCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RelativeRiskCalculatorPolars(AbstractRelativeRiskCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RelativeRiskCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ContingencyAnalysisCalculatorPolars(AbstractContingencyAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ContingencyAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CorrelationMatrixComputerPolars(AbstractCorrelationMatrixComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationMatrixComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class CorrelationPairExtractorPolars(AbstractCorrelationPairExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationPairExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class HighCorrelationFlagDetectorPolars(AbstractHighCorrelationFlagDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HighCorrelationFlagDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class CorrelationMatrixCalculatorPolars(AbstractCorrelationMatrixCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationMatrixCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SeriesStandardizerPolars(AbstractSeriesStandardizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SeriesStandardizerPandas()

    def standardize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.standardize(data, column, **kwargs)

class LaggedCorrelationComputerPolars(AbstractLaggedCorrelationComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LaggedCorrelationComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class PeakLagDetectorPolars(AbstractPeakLagDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PeakLagDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class CrossCorrelationCalculatorPolars(AbstractCrossCorrelationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CrossCorrelationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class LaggedDesignMatrixBuilderPolars(AbstractLaggedDesignMatrixBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LaggedDesignMatrixBuilderPandas()

    def build_restricted(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build_restricted(data, column, **kwargs)

    def build_unrestricted(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build_unrestricted(data, column, **kwargs)

class OLSResidualCalculatorPolars(AbstractOLSResidualCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = OLSResidualCalculatorPandas()

    def residual_ss(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.residual_ss(data, column, **kwargs)

class GrangerFStatisticCalculatorPolars(AbstractGrangerFStatisticCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GrangerFStatisticCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GrangerCausalityCalculatorPolars(AbstractGrangerCausalityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GrangerCausalityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AdditiveModelEvaluatorPolars(AbstractAdditiveModelEvaluator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AdditiveModelEvaluatorPandas()

    def evaluate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.evaluate(data, column, **kwargs)

class InteractionModelEvaluatorPolars(AbstractInteractionModelEvaluator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = InteractionModelEvaluatorPandas()

    def evaluate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.evaluate(data, column, **kwargs)

class InteractionGainClassifierPolars(AbstractInteractionGainClassifier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = InteractionGainClassifierPandas()

    def is_meaningful(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.is_meaningful(data, column, **kwargs)

class InteractionEffectsCalculatorPolars(AbstractInteractionEffectsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = InteractionEffectsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VIFRiskClassifierPolars(AbstractVIFRiskClassifier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = VIFRiskClassifierPandas()

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

class SingleFeatureVIFCalculatorPolars(AbstractSingleFeatureVIFCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SingleFeatureVIFCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MulticollinearityCalculatorPolars(AbstractMulticollinearityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MulticollinearityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TargetTypeDetectorPolars(AbstractTargetTypeDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TargetTypeDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class MIScoreNormalizerPolars(AbstractMIScoreNormalizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MIScoreNormalizerPandas()

    def normalize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class MutualInformationCalculatorPolars(AbstractMutualInformationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MutualInformationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ResidualExtractorPolars(AbstractResidualExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ResidualExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class PartialCorrelationCalculatorPolars(AbstractPartialCorrelationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PartialCorrelationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
