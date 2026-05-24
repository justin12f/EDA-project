"""Pandas statistics backends — `descriptive`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.descriptive.abstract import *

import statistics.descriptive.central_tendency as _mod_central_tendency
import statistics.descriptive.dispersion as _mod_dispersion
import statistics.descriptive.distribution as _mod_distribution
import statistics.descriptive.frequency as _mod_frequency
import statistics.descriptive.normality as _mod_normality
import statistics.descriptive.percentiles as _mod_percentiles
import statistics.descriptive.skewness_kurtosis as _mod_skewness_kurtosis
import statistics.descriptive.value_counts as _mod_value_counts

class MeanCalculatorPandas(AbstractMeanCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_central_tendency.MeanCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MedianCalculatorPandas(AbstractMedianCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_central_tendency.MedianCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ModeCalculatorPandas(AbstractModeCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_central_tendency.ModeCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TrimmedMeanCalculatorPandas(AbstractTrimmedMeanCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_central_tendency.TrimmedMeanCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CentralTendencyInterpreterPandas(AbstractCentralTendencyInterpreter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_central_tendency.CentralTendencyInterpreter()

    def interpret(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret(arr, **kwargs)

class CentralTendencyCalculatorPandas(AbstractCentralTendencyCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_central_tendency.CentralTendencyCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class VarianceCalculatorPandas(AbstractVarianceCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dispersion.VarianceCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class StandardDeviationCalculatorPandas(AbstractStandardDeviationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dispersion.StandardDeviationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RangeCalculatorPandas(AbstractRangeCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dispersion.RangeCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class IQRCalculatorPandas(AbstractIQRCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dispersion.IQRCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MADCalculatorPandas(AbstractMADCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dispersion.MADCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CoefficientOfVariationCalculatorPandas(AbstractCoefficientOfVariationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dispersion.CoefficientOfVariationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class DispersionCalculatorPandas(AbstractDispersionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_dispersion.DispersionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BimodalityDetectorPandas(AbstractBimodalityDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_distribution.BimodalityDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class TransformationAdvisorPandas(AbstractTransformationAdvisor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_distribution.TransformationAdvisor()

    def advise(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.advise(arr, **kwargs)

class DistributionFitterPandas(AbstractDistributionFitter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_distribution.DistributionFitter()

    def fit_all(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.fit_all(arr, **kwargs)

class DistributionClassifierPandas(AbstractDistributionClassifier[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_distribution.DistributionClassifier()

    def classify(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.classify(arr, **kwargs)

class BinCountSelectorPandas(AbstractBinCountSelector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_frequency.BinCountSelector()

    def select(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.select(arr, **kwargs)

class FrequencyTableBuilderPandas(AbstractFrequencyTableBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_frequency.FrequencyTableBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class FrequencyDistributionBuilderPandas(AbstractFrequencyDistributionBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_frequency.FrequencyDistributionBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class BaseNormalityTestPandas(AbstractBaseNormalityTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_normality.BaseNormalityTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class ShapiroWilkTestPandas(AbstractShapiroWilkTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_normality.ShapiroWilkTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class AndersonDarlingTestPandas(AbstractAndersonDarlingTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_normality.AndersonDarlingTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class KolmogorovSmirnovTestPandas(AbstractKolmogorovSmirnovTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_normality.KolmogorovSmirnovTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class NormalityTestSuitePandas(AbstractNormalityTestSuite[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_normality.NormalityTestSuite()

    def run(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.run(arr, **kwargs)

class PercentileOutlierDetectorPandas(AbstractPercentileOutlierDetector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_percentiles.PercentileOutlierDetector()

    def detect(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.detect(arr, **kwargs)

class PercentilesCalculatorPandas(AbstractPercentilesCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_percentiles.PercentilesCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SkewnessInterpreterPandas(AbstractSkewnessInterpreter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_skewness_kurtosis.SkewnessInterpreter()

    def interpret(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret(arr, **kwargs)

class KurtosisInterpreterPandas(AbstractKurtosisInterpreter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_skewness_kurtosis.KurtosisInterpreter()

    def interpret(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret(arr, **kwargs)

class SkewnessKurtosisCalculatorPandas(AbstractSkewnessKurtosisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_skewness_kurtosis.SkewnessKurtosisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ValueCountsCalculatorPandas(AbstractValueCountsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_value_counts.ValueCountsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
