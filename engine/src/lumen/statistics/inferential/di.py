"""Dependency injection container for the inferential statistics domain."""
from __future__ import annotations

from typing import Literal

from inferential.abstract.anova import AbstractANOVACalculator
from inferential.abstract.bootstrap import AbstractBootstrapEstimator
from inferential.abstract.chi_square import AbstractChiSquareCalculator
from inferential.abstract.confidence_intervals import AbstractConfidenceIntervalCalculator
from inferential.abstract.correlation_significance import AbstractCorrelationSignificanceCalculator
from inferential.abstract.effect_size import AbstractEffectSizeCalculator
from inferential.abstract.hypothesis_test import AbstractHypothesisTestSuite
from inferential.abstract.power_analysis import AbstractPowerAnalysisCalculator
from inferential.factory import InferentialStatisticsFactory

Backend = Literal["polars", "spark", "pandas"]


class InferentialDependencyContainer:
    def __init__(self, backend: Backend) -> None:
        self._backend: Backend = backend
        self._factory = InferentialStatisticsFactory

        self._anova_calc: AbstractANOVACalculator | None = None
        self._bootstrap_est: AbstractBootstrapEstimator | None = None
        self._chi_square_calc: AbstractChiSquareCalculator | None = None
        self._ci_calc: AbstractConfidenceIntervalCalculator | None = None
        self._correlation_calc: AbstractCorrelationSignificanceCalculator | None = None
        self._effect_size_calc: AbstractEffectSizeCalculator | None = None
        self._hypothesis_suite: AbstractHypothesisTestSuite | None = None
        self._power_calc: AbstractPowerAnalysisCalculator | None = None

    @property
    def backend(self) -> Backend:
        return self._backend

    def anova_calculator(self) -> AbstractANOVACalculator:
        if self._anova_calc is None:
            self._anova_calc = self._factory.create("anova_calculator", self._backend)
        return self._anova_calc

    def bootstrap_estimator(self) -> AbstractBootstrapEstimator:
        if self._bootstrap_est is None:
            self._bootstrap_est = self._factory.create("bootstrap_estimator", self._backend)
        return self._bootstrap_est

    def chi_square_calculator(self) -> AbstractChiSquareCalculator:
        if self._chi_square_calc is None:
            self._chi_square_calc = self._factory.create("chi_square_calculator", self._backend)
        return self._chi_square_calc

    def confidence_interval_calculator(self) -> AbstractConfidenceIntervalCalculator:
        if self._ci_calc is None:
            self._ci_calc = self._factory.create("confidence_interval_calculator", self._backend)
        return self._ci_calc

    def correlation_significance_calculator(self) -> AbstractCorrelationSignificanceCalculator:
        if self._correlation_calc is None:
            self._correlation_calc = self._factory.create("correlation_significance_calculator", self._backend)
        return self._correlation_calc

    def effect_size_calculator(self) -> AbstractEffectSizeCalculator:
        if self._effect_size_calc is None:
            self._effect_size_calc = self._factory.create("effect_size_calculator", self._backend)
        return self._effect_size_calc

    def hypothesis_test_suite(self) -> AbstractHypothesisTestSuite:
        if self._hypothesis_suite is None:
            self._hypothesis_suite = self._factory.create("hypothesis_test_suite", self._backend)
        return self._hypothesis_suite

    def power_analysis_calculator(self) -> AbstractPowerAnalysisCalculator:
        if self._power_calc is None:
            self._power_calc = self._factory.create("power_analysis_calculator", self._backend)
        return self._power_calc
