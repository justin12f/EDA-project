"""Factory for the descriptive statistics domain.

Registers all three backends (polars, spark, pandas) for every calculator
that the domain exposes.  The pandas backend delegates to the original
implementation classes so no existing logic is duplicated.
"""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class DescriptiveStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the descriptive statistics domain."""


def _register() -> None:
    from descriptive.backends import polars_impl as pl_impl
    from descriptive.backends import spark_impl as sp_impl

    # --- Polars ---
    DescriptiveStatisticsFactory.register("central_tendency_calculator", "polars", pl_impl.CentralTendencyCalculatorPolars)
    DescriptiveStatisticsFactory.register("dispersion_calculator",       "polars", pl_impl.DispersionCalculatorPolars)
    DescriptiveStatisticsFactory.register("distribution_classifier",     "polars", pl_impl.DistributionClassifierPolars)
    DescriptiveStatisticsFactory.register("frequency_distribution_builder", "polars", pl_impl.FrequencyDistributionBuilderPolars)
    DescriptiveStatisticsFactory.register("normality_test_suite",        "polars", pl_impl.NormalityTestSuitePolars)
    DescriptiveStatisticsFactory.register("percentiles_calculator",      "polars", pl_impl.PercentilesCalculatorPolars)
    DescriptiveStatisticsFactory.register("skewness_kurtosis_calculator","polars", pl_impl.SkewnessKurtosisCalculatorPolars)
    DescriptiveStatisticsFactory.register("value_counts_calculator",     "polars", pl_impl.ValueCountsCalculatorPolars)

    # --- Spark ---
    DescriptiveStatisticsFactory.register("central_tendency_calculator", "spark", sp_impl.CentralTendencyCalculatorSpark)
    DescriptiveStatisticsFactory.register("dispersion_calculator",       "spark", sp_impl.DispersionCalculatorSpark)
    DescriptiveStatisticsFactory.register("distribution_classifier",     "spark", sp_impl.DistributionClassifierSpark)
    DescriptiveStatisticsFactory.register("frequency_distribution_builder", "spark", sp_impl.FrequencyDistributionBuilderSpark)
    DescriptiveStatisticsFactory.register("normality_test_suite",        "spark", sp_impl.NormalityTestSuiteSpark)
    DescriptiveStatisticsFactory.register("percentiles_calculator",      "spark", sp_impl.PercentilesCalculatorSpark)
    DescriptiveStatisticsFactory.register("skewness_kurtosis_calculator","spark", sp_impl.SkewnessKurtosisCalculatorSpark)
    DescriptiveStatisticsFactory.register("value_counts_calculator",     "spark", sp_impl.ValueCountsCalculatorSpark)

    # --- Pandas (original implementations wrapped) ---
    from descriptive.backends import pandas_impl as pd_impl  # noqa: PLC0415
    DescriptiveStatisticsFactory.register("central_tendency_calculator", "pandas", pd_impl.CentralTendencyCalculatorPandas)
    DescriptiveStatisticsFactory.register("dispersion_calculator",       "pandas", pd_impl.DispersionCalculatorPandas)
    DescriptiveStatisticsFactory.register("distribution_classifier",     "pandas", pd_impl.DistributionClassifierPandas)
    DescriptiveStatisticsFactory.register("frequency_distribution_builder", "pandas", pd_impl.FrequencyDistributionBuilderPandas)
    DescriptiveStatisticsFactory.register("normality_test_suite",        "pandas", pd_impl.NormalityTestSuitePandas)
    DescriptiveStatisticsFactory.register("percentiles_calculator",      "pandas", pd_impl.PercentilesCalculatorPandas)
    DescriptiveStatisticsFactory.register("skewness_kurtosis_calculator","pandas", pd_impl.SkewnessKurtosisCalculatorPandas)
    DescriptiveStatisticsFactory.register("value_counts_calculator",     "pandas", pd_impl.ValueCountsCalculatorPandas)


_register()
