"""Pandas adapter backend for the inferential statistics domain."""
from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

from lumen.statistics.inferential.abstract.anova import AbstractANOVACalculator
from lumen.statistics.inferential.abstract.bootstrap import AbstractBootstrapEstimator
from lumen.statistics.inferential.abstract.chi_square import AbstractChiSquareCalculator
from lumen.statistics.inferential.abstract.confidence_intervals import AbstractConfidenceIntervalCalculator
from lumen.statistics.inferential.abstract.correlation_significance import AbstractCorrelationSignificanceCalculator
from lumen.statistics.inferential.abstract.effect_size import AbstractEffectSizeCalculator
from lumen.statistics.inferential.abstract.hypothesis_test import AbstractHypothesisTestSuite
from lumen.statistics.inferential.abstract.power_analysis import AbstractPowerAnalysisCalculator

from lumen.statistics.inferential.anova import OneWayAnovaCalculator
from lumen.statistics.inferential.bootstrap import BootstrapEstimator
from lumen.statistics.inferential.chi_square import ChiSquareCalculator
from lumen.statistics.inferential.confidence_intervals import ConfidenceIntervalCalculator
from lumen.statistics.inferential.correlation import CorrelationCalculator
from lumen.statistics.inferential.effect_size import EffectSizeCalculator
from lumen.statistics.inferential.hypothesis_test import HypothesisTestSuite
from lumen.statistics.inferential.power_analysis import PowerAnalysisCalculator
from lumen.statistics.core.frame_extract import column_to_numpy


class ANOVACalculatorPandas(AbstractANOVACalculator):
    def __init__(self) -> None:
        self._impl = OneWayAnovaCalculator()

    def calculate(
        self,
        data: Any,
        value_column: str,
        group_column: str,
        significance_level: float = 0.05,
        run_post_hoc: bool = True,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        groups = {str(name): group[value_column].values for name, group in df.groupby(group_column)}
        return self._impl.calculate(groups, significance_level=significance_level, run_post_hoc=run_post_hoc)


class BootstrapEstimatorPandas(AbstractBootstrapEstimator):
    def __init__(self) -> None:
        self._impl = BootstrapEstimator()

    def estimate(
        self,
        data: Any,
        column: str,
        statistic_expr: Any,
        n_iterations: int = 5_000,
        confidence_level: float = 0.95,
        random_seed: int | None = 42,
    ) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.estimate(
            data=arr,
            statistic=statistic_expr,
            n_iterations=n_iterations,
            confidence_level=confidence_level,
            random_seed=random_seed
        )


class ChiSquareCalculatorPandas(AbstractChiSquareCalculator):
    def __init__(self) -> None:
        self._impl = ChiSquareCalculator()

    def calculate(
        self,
        data: Any,
        column1: str,
        column2: str | None = None,
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if column2 is None:
            counts = df[column1].value_counts().values
            return self._impl.calculate("goodness_of_fit", observed_frequencies=counts, significance_level=significance_level)
        else:
            crosstab = pd.crosstab(df[column1], df[column2]).values
            return self._impl.calculate("independence", contingency_table=crosstab, significance_level=significance_level)


class ConfidenceIntervalCalculatorPandas(AbstractConfidenceIntervalCalculator):
    def __init__(self) -> None:
        self._impl = ConfidenceIntervalCalculator()

    def calculate(
        self,
        data: Any,
        column: str,
        confidence_level: float = 0.95,
        method: str = "t",
    ) -> dict[str, Any]:
        arr = column_to_numpy(data, column)
        return self._impl.calculate("mean", data=arr, confidence_level=confidence_level)


class CorrelationSignificanceCalculatorPandas(AbstractCorrelationSignificanceCalculator):
    def __init__(self) -> None:
        self._impl = CorrelationCalculator()

    def calculate(
        self,
        data: Any,
        column1: str,
        column2: str,
        method: str = "pearson",
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        arr1 = column_to_numpy(data, column1)
        arr2 = column_to_numpy(data, column2)
        return self._impl.calculate(method, arr1, arr2, significance_level=significance_level)


class EffectSizeCalculatorPandas(AbstractEffectSizeCalculator):
    def __init__(self) -> None:
        self._impl = EffectSizeCalculator()

    def calculate(
        self,
        data: Any,
        value_column: str,
        group_column: str,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        groups = [group[value_column].values for name, group in df.groupby(group_column)]
        if len(groups) != 2:
            raise ValueError("Effect size requires exactly 2 groups.")
        return self._impl.calculate("cohens_d", groups[0], groups[1])


class HypothesisTestSuitePandas(AbstractHypothesisTestSuite):
    def __init__(self) -> None:
        self._impl = HypothesisTestSuite()

    def run(
        self,
        data: Any,
        value_column: str,
        group_column: str,
        test_type: str = "t_test_ind",
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        groups = [group[value_column].values for name, group in df.groupby(group_column)]
        if len(groups) != 2:
            raise ValueError("Hypothesis test requires exactly 2 groups.")
        
        test_key = "t_test" if "t_test" in test_type else "mann_whitney"
        return self._impl.run(groups[0], groups[1], test=test_key, significance_level=significance_level)


class PowerAnalysisCalculatorPandas(AbstractPowerAnalysisCalculator):
    def __init__(self) -> None:
        self._impl = PowerAnalysisCalculator()

    def calculate(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float | None = 0.8,
        n: int | None = None,
        test_type: str = "t_test_ind",
    ) -> dict[str, Any]:
        return self._impl.calculate(
            test_type=test_type,
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            n=n
        )
