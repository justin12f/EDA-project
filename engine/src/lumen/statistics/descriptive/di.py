"""Dependency injection container for the descriptive statistics domain.

Usage
-----
    from lumen.statistics.descriptive.di import DescriptiveDependencyContainer

    # Select backend once at startup
    di = DescriptiveDependencyContainer(backend="polars")

    # Use typed convenience accessors — no string keys needed downstream
    calc = di.central_tendency_calculator()
    result = calc.calculate(data=my_dataframe, column="age")
"""
from __future__ import annotations

from typing import Literal

from lumen.statistics.descriptive.abstract.central_tendency import AbstractCentralTendencyCalculator
from lumen.statistics.descriptive.abstract.dispersion import AbstractDispersionCalculator
from lumen.statistics.descriptive.abstract.distribution import AbstractDistributionClassifier
from lumen.statistics.descriptive.abstract.frequency import AbstractFrequencyDistributionBuilder
from lumen.statistics.descriptive.abstract.normality import AbstractNormalityTestSuite
from lumen.statistics.descriptive.abstract.percentiles import AbstractPercentilesCalculator
from lumen.statistics.descriptive.abstract.skewness_kurtosis import AbstractSkewnessKurtosisCalculator
from lumen.statistics.descriptive.abstract.value_counts import AbstractValueCountsCalculator
from lumen.statistics.descriptive.factory import DescriptiveStatisticsFactory

Backend = Literal["polars", "spark", "pandas"]


class DescriptiveDependencyContainer:
    """Resolves and caches all descriptive statistics calculator instances.

    All calculators are instantiated lazily on first access and cached
    as instance attributes.  Thread-safety is not required here because
    calculator instances are stateless.

    Parameters
    ----------
    backend:
        One of ``"polars"``, ``"spark"``, or ``"pandas"``.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend: Backend = backend
        self._factory = DescriptiveStatisticsFactory

        # Lazily-populated cache
        self._central_tendency_calc: AbstractCentralTendencyCalculator | None = None
        self._dispersion_calc: AbstractDispersionCalculator | None = None
        self._distribution_classifier: AbstractDistributionClassifier | None = None
        self._frequency_builder: AbstractFrequencyDistributionBuilder | None = None
        self._normality_suite: AbstractNormalityTestSuite | None = None
        self._percentiles_calc: AbstractPercentilesCalculator | None = None
        self._skewness_kurtosis_calc: AbstractSkewnessKurtosisCalculator | None = None
        self._value_counts_calc: AbstractValueCountsCalculator | None = None

    @property
    def backend(self) -> Backend:
        """Active backend name."""
        return self._backend

    # ------------------------------------------------------------------
    # Typed accessor methods
    # ------------------------------------------------------------------

    def central_tendency_calculator(self) -> AbstractCentralTendencyCalculator:
        """Return the cached CentralTendencyCalculator for the active backend."""
        if self._central_tendency_calc is None:
            self._central_tendency_calc = self._factory.create(
                "central_tendency_calculator", self._backend
            )
        return self._central_tendency_calc

    def dispersion_calculator(self) -> AbstractDispersionCalculator:
        """Return the cached DispersionCalculator for the active backend."""
        if self._dispersion_calc is None:
            self._dispersion_calc = self._factory.create(
                "dispersion_calculator", self._backend
            )
        return self._dispersion_calc

    def distribution_classifier(self) -> AbstractDistributionClassifier:
        """Return the cached DistributionClassifier for the active backend."""
        if self._distribution_classifier is None:
            self._distribution_classifier = self._factory.create(
                "distribution_classifier", self._backend
            )
        return self._distribution_classifier

    def frequency_distribution_builder(self) -> AbstractFrequencyDistributionBuilder:
        """Return the cached FrequencyDistributionBuilder for the active backend."""
        if self._frequency_builder is None:
            self._frequency_builder = self._factory.create(
                "frequency_distribution_builder", self._backend
            )
        return self._frequency_builder

    def normality_test_suite(self) -> AbstractNormalityTestSuite:
        """Return the cached NormalityTestSuite for the active backend."""
        if self._normality_suite is None:
            self._normality_suite = self._factory.create(
                "normality_test_suite", self._backend
            )
        return self._normality_suite

    def percentiles_calculator(self) -> AbstractPercentilesCalculator:
        """Return the cached PercentilesCalculator for the active backend."""
        if self._percentiles_calc is None:
            self._percentiles_calc = self._factory.create(
                "percentiles_calculator", self._backend
            )
        return self._percentiles_calc

    def skewness_kurtosis_calculator(self) -> AbstractSkewnessKurtosisCalculator:
        """Return the cached SkewnessKurtosisCalculator for the active backend."""
        if self._skewness_kurtosis_calc is None:
            self._skewness_kurtosis_calc = self._factory.create(
                "skewness_kurtosis_calculator", self._backend
            )
        return self._skewness_kurtosis_calc

    def value_counts_calculator(self) -> AbstractValueCountsCalculator:
        """Return the cached ValueCountsCalculator for the active backend."""
        if self._value_counts_calc is None:
            self._value_counts_calc = self._factory.create(
                "value_counts_calculator", self._backend
            )
        return self._value_counts_calc
