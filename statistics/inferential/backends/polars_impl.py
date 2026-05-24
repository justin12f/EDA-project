"""Polars statistics backends — `inferential`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.inferential.backends import pandas_impl
from statistics.inferential.backends.pandas_impl import *

from statistics.inferential.abstract import *

class TukeyHSDPostHocPolars(AbstractTukeyHSDPostHoc[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TukeyHSDPostHocPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class OneWayAnovaCalculatorPolars(AbstractOneWayAnovaCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = OneWayAnovaCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BootstrapSamplerPolars(AbstractBootstrapSampler[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BootstrapSamplerPandas()

    def generate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.generate(data, column, **kwargs)

class BootstrapStatisticEstimatorPolars(AbstractBootstrapStatisticEstimator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BootstrapStatisticEstimatorPandas()

    def estimate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class PercentilesBootstrapCIPolars(AbstractPercentilesBootstrapCI[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PercentilesBootstrapCIPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BootstrapEstimatorPolars(AbstractBootstrapEstimator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BootstrapEstimatorPandas()

    def estimate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class ContingencyTableBuilderPolars(AbstractContingencyTableBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ContingencyTableBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class CramersVCalculatorPolars(AbstractCramersVCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CramersVCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ChiSquareTestCalculatorPolars(AbstractChiSquareTestCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ChiSquareTestCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseConfidenceIntervalPolars(AbstractBaseConfidenceInterval[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseConfidenceIntervalPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MeanConfidenceIntervalPolars(AbstractMeanConfidenceInterval[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanConfidenceIntervalPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ProportionConfidenceIntervalPolars(AbstractProportionConfidenceInterval[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ProportionConfidenceIntervalPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MeanDifferenceConfidenceIntervalPolars(AbstractMeanDifferenceConfidenceInterval[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanDifferenceConfidenceIntervalPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ConfidenceIntervalCalculatorPolars(AbstractConfidenceIntervalCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ConfidenceIntervalCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CorrelationInterpreterPolars(AbstractCorrelationInterpreter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationInterpreterPandas()

    def interpret(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class FisherZTransformerPolars(AbstractFisherZTransformer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FisherZTransformerPandas()

    def confidence_interval(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.confidence_interval(data, column, **kwargs)

class CorrelationSignificanceCalculatorPolars(AbstractCorrelationSignificanceCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationSignificanceCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EffectSizeInterpreterPolars(AbstractEffectSizeInterpreter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EffectSizeInterpreterPandas()

    def interpret_cohens_d(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret_cohens_d(data, column, **kwargs)

    def interpret_cramers_v(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret_cramers_v(data, column, **kwargs)

    def interpret_eta_squared(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret_eta_squared(data, column, **kwargs)

class CohensDCalculatorPolars(AbstractCohensDCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CohensDCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EtaSquaredCalculatorPolars(AbstractEtaSquaredCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EtaSquaredCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EffectSizeCalculatorPolars(AbstractEffectSizeCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EffectSizeCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class HypothesisInterpreterPolars(AbstractHypothesisInterpreter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HypothesisInterpreterPandas()

    def interpret(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class BaseHypothesisTestPolars(AbstractBaseHypothesisTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseHypothesisTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class TTestPolars(AbstractTTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class MannWhitneyTestPolars(AbstractMannWhitneyTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MannWhitneyTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class WilcoxonTestPolars(AbstractWilcoxonTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = WilcoxonTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class HypothesisTestSuitePolars(AbstractHypothesisTestSuite[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HypothesisTestSuitePandas()

    def run(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.run(data, column, **kwargs)

class MinimumSampleSizeCalculatorPolars(AbstractMinimumSampleSizeCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MinimumSampleSizeCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ObservedPowerCalculatorPolars(AbstractObservedPowerCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ObservedPowerCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PowerAnalysisCalculatorPolars(AbstractPowerAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PowerAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
