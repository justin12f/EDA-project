"""Pandas statistics backends — `inferential`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.inferential.abstract import *

import statistics.inferential.anova as _mod_anova
import statistics.inferential.bootstrap as _mod_bootstrap
import statistics.inferential.chi_square as _mod_chi_square
import statistics.inferential.confidence_intervals as _mod_confidence_intervals
import statistics.inferential.correlation_significance as _mod_correlation_significance
import statistics.inferential.effect_size as _mod_effect_size
import statistics.inferential.hypothesis_test as _mod_hypothesis_test
import statistics.inferential.power_analysis as _mod_power_analysis

class TukeyHSDPostHocPandas(AbstractTukeyHSDPostHoc[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_anova.TukeyHSDPostHoc()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class OneWayAnovaCalculatorPandas(AbstractOneWayAnovaCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_anova.OneWayAnovaCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BootstrapSamplerPandas(AbstractBootstrapSampler[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_bootstrap.BootstrapSampler()

    def generate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.generate(arr, **kwargs)

class BootstrapStatisticEstimatorPandas(AbstractBootstrapStatisticEstimator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_bootstrap.BootstrapStatisticEstimator()

    def estimate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.estimate(arr, **kwargs)

class PercentilesBootstrapCIPandas(AbstractPercentilesBootstrapCI[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_bootstrap.PercentilesBootstrapCI()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BootstrapEstimatorPandas(AbstractBootstrapEstimator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_bootstrap.BootstrapEstimator()

    def estimate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.estimate(arr, **kwargs)

class ContingencyTableBuilderPandas(AbstractContingencyTableBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_chi_square.ContingencyTableBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class CramersVCalculatorPandas(AbstractCramersVCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_chi_square.CramersVCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ChiSquareTestCalculatorPandas(AbstractChiSquareTestCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_chi_square.ChiSquareTestCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BaseConfidenceIntervalPandas(AbstractBaseConfidenceInterval[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_confidence_intervals.BaseConfidenceInterval()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MeanConfidenceIntervalPandas(AbstractMeanConfidenceInterval[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_confidence_intervals.MeanConfidenceInterval()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ProportionConfidenceIntervalPandas(AbstractProportionConfidenceInterval[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_confidence_intervals.ProportionConfidenceInterval()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MeanDifferenceConfidenceIntervalPandas(AbstractMeanDifferenceConfidenceInterval[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_confidence_intervals.MeanDifferenceConfidenceInterval()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ConfidenceIntervalCalculatorPandas(AbstractConfidenceIntervalCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_confidence_intervals.ConfidenceIntervalCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CorrelationInterpreterPandas(AbstractCorrelationInterpreter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_correlation_significance.CorrelationInterpreter()

    def interpret(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret(arr, **kwargs)

class FisherZTransformerPandas(AbstractFisherZTransformer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_correlation_significance.FisherZTransformer()

    def confidence_interval(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.confidence_interval(arr, **kwargs)

class CorrelationSignificanceCalculatorPandas(AbstractCorrelationSignificanceCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_correlation_significance.CorrelationSignificanceCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class EffectSizeInterpreterPandas(AbstractEffectSizeInterpreter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_effect_size.EffectSizeInterpreter()

    def interpret_cohens_d(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret_cohens_d(arr, **kwargs)

    def interpret_cramers_v(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret_cramers_v(arr, **kwargs)

    def interpret_eta_squared(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret_eta_squared(arr, **kwargs)

class CohensDCalculatorPandas(AbstractCohensDCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_effect_size.CohensDCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class EtaSquaredCalculatorPandas(AbstractEtaSquaredCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_effect_size.EtaSquaredCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class EffectSizeCalculatorPandas(AbstractEffectSizeCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_effect_size.EffectSizeCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class HypothesisInterpreterPandas(AbstractHypothesisInterpreter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hypothesis_test.HypothesisInterpreter()

    def interpret(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.interpret(arr, **kwargs)

class BaseHypothesisTestPandas(AbstractBaseHypothesisTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hypothesis_test.BaseHypothesisTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class TTestPandas(AbstractTTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hypothesis_test.TTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class MannWhitneyTestPandas(AbstractMannWhitneyTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hypothesis_test.MannWhitneyTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class WilcoxonTestPandas(AbstractWilcoxonTest[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hypothesis_test.WilcoxonTest()

    def test(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.test(arr, **kwargs)

class HypothesisTestSuitePandas(AbstractHypothesisTestSuite[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_hypothesis_test.HypothesisTestSuite()

    def run(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.run(arr, **kwargs)

class MinimumSampleSizeCalculatorPandas(AbstractMinimumSampleSizeCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_power_analysis.MinimumSampleSizeCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ObservedPowerCalculatorPandas(AbstractObservedPowerCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_power_analysis.ObservedPowerCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class PowerAnalysisCalculatorPandas(AbstractPowerAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_power_analysis.PowerAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
