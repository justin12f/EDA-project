"""Spark statistics backends — `inferential`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.inferential.abstract import *

from statistics.inferential.backends import pandas_impl
from statistics.inferential.backends.pandas_impl import *

class TukeyHSDPostHocSpark(AbstractTukeyHSDPostHoc[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TukeyHSDPostHocPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class OneWayAnovaCalculatorSpark(AbstractOneWayAnovaCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = OneWayAnovaCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BootstrapSamplerSpark(AbstractBootstrapSampler[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BootstrapSamplerPandas()

    def generate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.generate(data, column, **kwargs)

class BootstrapStatisticEstimatorSpark(AbstractBootstrapStatisticEstimator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BootstrapStatisticEstimatorPandas()

    def estimate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class PercentilesBootstrapCISpark(AbstractPercentilesBootstrapCI[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PercentilesBootstrapCIPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BootstrapEstimatorSpark(AbstractBootstrapEstimator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BootstrapEstimatorPandas()

    def estimate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class ContingencyTableBuilderSpark(AbstractContingencyTableBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ContingencyTableBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class CramersVCalculatorSpark(AbstractCramersVCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CramersVCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ChiSquareTestCalculatorSpark(AbstractChiSquareTestCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ChiSquareTestCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseConfidenceIntervalSpark(AbstractBaseConfidenceInterval[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseConfidenceIntervalPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MeanConfidenceIntervalSpark(AbstractMeanConfidenceInterval[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanConfidenceIntervalPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ProportionConfidenceIntervalSpark(AbstractProportionConfidenceInterval[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ProportionConfidenceIntervalPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MeanDifferenceConfidenceIntervalSpark(AbstractMeanDifferenceConfidenceInterval[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanDifferenceConfidenceIntervalPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ConfidenceIntervalCalculatorSpark(AbstractConfidenceIntervalCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ConfidenceIntervalCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CorrelationInterpreterSpark(AbstractCorrelationInterpreter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationInterpreterPandas()

    def interpret(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class FisherZTransformerSpark(AbstractFisherZTransformer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FisherZTransformerPandas()

    def confidence_interval(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.confidence_interval(data, column, **kwargs)

class CorrelationSignificanceCalculatorSpark(AbstractCorrelationSignificanceCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CorrelationSignificanceCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EffectSizeInterpreterSpark(AbstractEffectSizeInterpreter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EffectSizeInterpreterPandas()

    def interpret_cohens_d(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret_cohens_d(data, column, **kwargs)

    def interpret_cramers_v(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret_cramers_v(data, column, **kwargs)

    def interpret_eta_squared(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret_eta_squared(data, column, **kwargs)

class CohensDCalculatorSpark(AbstractCohensDCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CohensDCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EtaSquaredCalculatorSpark(AbstractEtaSquaredCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EtaSquaredCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class EffectSizeCalculatorSpark(AbstractEffectSizeCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EffectSizeCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class HypothesisInterpreterSpark(AbstractHypothesisInterpreter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HypothesisInterpreterPandas()

    def interpret(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class BaseHypothesisTestSpark(AbstractBaseHypothesisTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseHypothesisTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class TTestSpark(AbstractTTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class MannWhitneyTestSpark(AbstractMannWhitneyTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MannWhitneyTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class WilcoxonTestSpark(AbstractWilcoxonTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = WilcoxonTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class HypothesisTestSuiteSpark(AbstractHypothesisTestSuite[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HypothesisTestSuitePandas()

    def run(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.run(data, column, **kwargs)

class MinimumSampleSizeCalculatorSpark(AbstractMinimumSampleSizeCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MinimumSampleSizeCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ObservedPowerCalculatorSpark(AbstractObservedPowerCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ObservedPowerCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PowerAnalysisCalculatorSpark(AbstractPowerAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PowerAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
