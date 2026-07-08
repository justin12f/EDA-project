"""Pandas adapter backend for the descriptive statistics domain.

These thin wrapper classes adapt the existing pandas/numpy-based
implementations to the abstract calculator interfaces.  The original
source files are never modified — this module is the only bridge between
the new abstract layer and the legacy implementation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from descriptive.abstract.central_tendency import AbstractCentralTendencyCalculator
from descriptive.abstract.dispersion import AbstractDispersionCalculator
from descriptive.abstract.distribution import AbstractDistributionClassifier
from descriptive.abstract.frequency import AbstractFrequencyDistributionBuilder
from descriptive.abstract.normality import AbstractNormalityTestSuite
from descriptive.abstract.percentiles import AbstractPercentilesCalculator
from descriptive.abstract.skewness_kurtosis import AbstractSkewnessKurtosisCalculator
from descriptive.abstract.value_counts import AbstractValueCountsCalculator

# Import original implementations
from descriptive.central_tendency import CentralTendencyCalculator
from descriptive.dispersion import DispersionCalculator
from descriptive.distribution import DistributionClassifier
from descriptive.frequency import FrequencyDistributionBuilder
from descriptive.normality import NormalityTestSuite
from descriptive.percentiles import PercentilesCalculator
from descriptive.skewness_kurtosis import SkewnessKurtosisCalculator
from descriptive.value_counts import ValueCountsCalculator
from core.frame_extract import column_to_numpy


class CentralTendencyCalculatorPandas(AbstractCentralTendencyCalculator):
    """Pandas adapter — delegates to original CentralTendencyCalculator."""

    def __init__(self) -> None:
        self._impl = CentralTendencyCalculator()

    def calculate(self, data: Any, column: str, trim_proportion: float = 0.1) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.calculate(arr, trim_proportion=trim_proportion)


class DispersionCalculatorPandas(AbstractDispersionCalculator):
    """Pandas adapter — delegates to original DispersionCalculator."""

    def __init__(self) -> None:
        self._impl = DispersionCalculator()

    def calculate(self, data: Any, column: str, ddof: int = 1) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.calculate(arr, ddof=ddof)


class DistributionClassifierPandas(AbstractDistributionClassifier):
    """Pandas adapter — delegates to original DistributionClassifier."""

    def __init__(self) -> None:
        self._impl = DistributionClassifier()

    def classify(self, data: Any, column: str) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.classify(arr)


class FrequencyDistributionBuilderPandas(AbstractFrequencyDistributionBuilder):
    """Pandas adapter — delegates to original FrequencyDistributionBuilder."""

    def __init__(self) -> None:
        self._impl = FrequencyDistributionBuilder()

    def build(self, data: Any, column: str, n_bins: int | None = None, bin_method: str = "auto") -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.build(arr, n_bins=n_bins, bin_method=bin_method)


class NormalityTestSuitePandas(AbstractNormalityTestSuite):
    """Pandas adapter — delegates to original NormalityTestSuite."""

    def __init__(self) -> None:
        self._impl = NormalityTestSuite()

    def run(self, data: Any, column: str, significance_level: float = 0.05) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.run(arr, significance_level=significance_level)


class PercentilesCalculatorPandas(AbstractPercentilesCalculator):
    """Pandas adapter — delegates to original PercentilesCalculator."""

    def __init__(self) -> None:
        self._impl = PercentilesCalculator()

    def calculate(self, data: Any, column: str, percentiles: list[int] | None = None, outlier_bounds: tuple[int, int] | None = (1, 99)) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.calculate(arr, percentiles=percentiles, outlier_bounds=outlier_bounds)


class SkewnessKurtosisCalculatorPandas(AbstractSkewnessKurtosisCalculator):
    """Pandas adapter — delegates to original SkewnessKurtosisCalculator."""

    def __init__(self) -> None:
        self._impl = SkewnessKurtosisCalculator()

    def calculate(self, data: Any, column: str, bias: bool = True) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.calculate(arr, bias=bias)


class ValueCountsCalculatorPandas(AbstractValueCountsCalculator):
    """Pandas adapter — delegates to original ValueCountsCalculator."""

    def __init__(self) -> None:
        self._impl = ValueCountsCalculator()

    def calculate(self, data: Any, column: str, top_n: int | None = None, include_missing: bool = True) -> dict[str, Any]:
        if isinstance(data, pd.DataFrame):
            series = data[column]
        else:
            series = pd.Series(column_to_numpy(data, column))
        return self._impl.calculate(series, top_n=top_n, include_missing=include_missing)
