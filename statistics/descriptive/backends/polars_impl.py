"""Polars statistics backends — `descriptive`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.descriptive.backends import pandas_impl
from statistics.descriptive.backends.pandas_impl import *

from statistics.descriptive.abstract import *

class MeanCalculatorPolars(AbstractMeanCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        frame = eager(data)
        return float(frame.select(pl.col(column).mean()).item())

class MedianCalculatorPolars(AbstractMedianCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MedianCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        frame = eager(data)
        return float(frame.select(pl.col(column).median()).item())

class ModeCalculatorPolars(AbstractModeCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ModeCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TrimmedMeanCalculatorPolars(AbstractTrimmedMeanCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TrimmedMeanCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CentralTendencyInterpreterPolars(AbstractCentralTendencyInterpreter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralTendencyInterpreterPandas()

    def interpret(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class CentralTendencyCalculatorPolars(AbstractCentralTendencyCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralTendencyCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class VarianceCalculatorPolars(AbstractVarianceCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = VarianceCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        frame = eager(data)
        return float(frame.select(pl.col(column).var()).item())

class StandardDeviationCalculatorPolars(AbstractStandardDeviationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = StandardDeviationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        frame = eager(data)
        return float(frame.select(pl.col(column).std()).item())

class RangeCalculatorPolars(AbstractRangeCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RangeCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class IQRCalculatorPolars(AbstractIQRCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = IQRCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MADCalculatorPolars(AbstractMADCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MADCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CoefficientOfVariationCalculatorPolars(AbstractCoefficientOfVariationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CoefficientOfVariationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DispersionCalculatorPolars(AbstractDispersionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DispersionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BimodalityDetectorPolars(AbstractBimodalityDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BimodalityDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class TransformationAdvisorPolars(AbstractTransformationAdvisor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TransformationAdvisorPandas()

    def advise(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.advise(data, column, **kwargs)

class DistributionFitterPolars(AbstractDistributionFitter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DistributionFitterPandas()

    def fit_all(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.fit_all(data, column, **kwargs)

class DistributionClassifierPolars(AbstractDistributionClassifier[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DistributionClassifierPandas()

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

class BinCountSelectorPolars(AbstractBinCountSelector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BinCountSelectorPandas()

    def select(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.select(data, column, **kwargs)

class FrequencyTableBuilderPolars(AbstractFrequencyTableBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FrequencyTableBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class FrequencyDistributionBuilderPolars(AbstractFrequencyDistributionBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FrequencyDistributionBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class BaseNormalityTestPolars(AbstractBaseNormalityTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseNormalityTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class ShapiroWilkTestPolars(AbstractShapiroWilkTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ShapiroWilkTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class AndersonDarlingTestPolars(AbstractAndersonDarlingTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AndersonDarlingTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class KolmogorovSmirnovTestPolars(AbstractKolmogorovSmirnovTest[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = KolmogorovSmirnovTestPandas()

    def test(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.test(data, column, **kwargs)

class NormalityTestSuitePolars(AbstractNormalityTestSuite[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NormalityTestSuitePandas()

    def run(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.run(data, column, **kwargs)

class PercentileOutlierDetectorPolars(AbstractPercentileOutlierDetector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PercentileOutlierDetectorPandas()

    def detect(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.detect(data, column, **kwargs)

class PercentilesCalculatorPolars(AbstractPercentilesCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PercentilesCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SkewnessInterpreterPolars(AbstractSkewnessInterpreter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SkewnessInterpreterPandas()

    def interpret(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class KurtosisInterpreterPolars(AbstractKurtosisInterpreter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = KurtosisInterpreterPandas()

    def interpret(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.interpret(data, column, **kwargs)

class SkewnessKurtosisCalculatorPolars(AbstractSkewnessKurtosisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SkewnessKurtosisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ValueCountsCalculatorPolars(AbstractValueCountsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ValueCountsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
