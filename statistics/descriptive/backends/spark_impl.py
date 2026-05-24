"""Spark statistics backends — `descriptive`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.descriptive.abstract import *

from statistics.descriptive.backends import pandas_impl
from statistics.descriptive.backends.pandas_impl import *

class MeanCalculatorSpark(AbstractMeanCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        row = data.select(F.mean(column).alias("v")).collect()[0]
        return float(row["v"])

class MedianCalculatorSpark(AbstractMedianCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MedianCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        row = data.select(F.expr(f"percentile_approx(`{column}`, 0.5)").alias("v")).collect()[0]
        return float(row["v"])

class ModeCalculatorSpark(AbstractModeCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ModeCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TrimmedMeanCalculatorSpark(AbstractTrimmedMeanCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TrimmedMeanCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CentralTendencyInterpreterSpark(AbstractCentralTendencyInterpreter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralTendencyInterpreterPandas()

    def interpret(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class CentralTendencyCalculatorSpark(AbstractCentralTendencyCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralTendencyCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VarianceCalculatorSpark(AbstractVarianceCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = VarianceCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        row = data.select(F.variance(column).alias("v")).collect()[0]
        return float(row["v"])

class StandardDeviationCalculatorSpark(AbstractStandardDeviationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = StandardDeviationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RangeCalculatorSpark(AbstractRangeCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RangeCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class IQRCalculatorSpark(AbstractIQRCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = IQRCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MADCalculatorSpark(AbstractMADCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MADCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CoefficientOfVariationCalculatorSpark(AbstractCoefficientOfVariationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CoefficientOfVariationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DispersionCalculatorSpark(AbstractDispersionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DispersionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BimodalityDetectorSpark(AbstractBimodalityDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BimodalityDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class TransformationAdvisorSpark(AbstractTransformationAdvisor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TransformationAdvisorPandas()

    def advise(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.advise(data, column, **kwargs)

class DistributionFitterSpark(AbstractDistributionFitter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DistributionFitterPandas()

    def fit_all(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.fit_all(data, column, **kwargs)

class DistributionClassifierSpark(AbstractDistributionClassifier[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DistributionClassifierPandas()

    def classify(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.classify(data, column, **kwargs)

class BinCountSelectorSpark(AbstractBinCountSelector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BinCountSelectorPandas()

    def select(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class FrequencyTableBuilderSpark(AbstractFrequencyTableBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FrequencyTableBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class FrequencyDistributionBuilderSpark(AbstractFrequencyDistributionBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FrequencyDistributionBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class BaseNormalityTestSpark(AbstractBaseNormalityTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseNormalityTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class ShapiroWilkTestSpark(AbstractShapiroWilkTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ShapiroWilkTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class AndersonDarlingTestSpark(AbstractAndersonDarlingTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AndersonDarlingTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class KolmogorovSmirnovTestSpark(AbstractKolmogorovSmirnovTest[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = KolmogorovSmirnovTestPandas()

    def test(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class NormalityTestSuiteSpark(AbstractNormalityTestSuite[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NormalityTestSuitePandas()

    def run(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.run(data, column, **kwargs)

class PercentileOutlierDetectorSpark(AbstractPercentileOutlierDetector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PercentileOutlierDetectorPandas()

    def detect(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class PercentilesCalculatorSpark(AbstractPercentilesCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PercentilesCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SkewnessInterpreterSpark(AbstractSkewnessInterpreter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SkewnessInterpreterPandas()

    def interpret(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class KurtosisInterpreterSpark(AbstractKurtosisInterpreter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = KurtosisInterpreterPandas()

    def interpret(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class SkewnessKurtosisCalculatorSpark(AbstractSkewnessKurtosisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SkewnessKurtosisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ValueCountsCalculatorSpark(AbstractValueCountsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ValueCountsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
