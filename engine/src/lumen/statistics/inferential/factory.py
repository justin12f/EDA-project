"""Factory for the inferential statistics domain."""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class InferentialStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the inferential statistics domain."""


def _register() -> None:
    from inferential.backends import polars_impl as pl_impl
    from inferential.backends import spark_impl as sp_impl
    from inferential.backends import pandas_impl as pd_impl

    # --- Polars ---
    InferentialStatisticsFactory.register("anova_calculator", "polars", pl_impl.ANOVACalculatorPolars)
    InferentialStatisticsFactory.register("bootstrap_estimator", "polars", pl_impl.BootstrapEstimatorPolars)
    InferentialStatisticsFactory.register("chi_square_calculator", "polars", pl_impl.ChiSquareCalculatorPolars)
    InferentialStatisticsFactory.register("confidence_interval_calculator", "polars", pl_impl.ConfidenceIntervalCalculatorPolars)
    InferentialStatisticsFactory.register("correlation_significance_calculator", "polars", pl_impl.CorrelationSignificanceCalculatorPolars)
    InferentialStatisticsFactory.register("effect_size_calculator", "polars", pl_impl.EffectSizeCalculatorPolars)
    InferentialStatisticsFactory.register("hypothesis_test_suite", "polars", pl_impl.HypothesisTestSuitePolars)
    InferentialStatisticsFactory.register("power_analysis_calculator", "polars", pl_impl.PowerAnalysisCalculatorPolars)

    # --- Spark ---
    InferentialStatisticsFactory.register("anova_calculator", "spark", sp_impl.ANOVACalculatorSpark)
    InferentialStatisticsFactory.register("bootstrap_estimator", "spark", sp_impl.BootstrapEstimatorSpark)
    InferentialStatisticsFactory.register("chi_square_calculator", "spark", sp_impl.ChiSquareCalculatorSpark)
    InferentialStatisticsFactory.register("confidence_interval_calculator", "spark", sp_impl.ConfidenceIntervalCalculatorSpark)
    InferentialStatisticsFactory.register("correlation_significance_calculator", "spark", sp_impl.CorrelationSignificanceCalculatorSpark)
    InferentialStatisticsFactory.register("effect_size_calculator", "spark", sp_impl.EffectSizeCalculatorSpark)
    InferentialStatisticsFactory.register("hypothesis_test_suite", "spark", sp_impl.HypothesisTestSuiteSpark)
    InferentialStatisticsFactory.register("power_analysis_calculator", "spark", sp_impl.PowerAnalysisCalculatorSpark)

    # --- Pandas ---
    InferentialStatisticsFactory.register("anova_calculator", "pandas", pd_impl.ANOVACalculatorPandas)
    InferentialStatisticsFactory.register("bootstrap_estimator", "pandas", pd_impl.BootstrapEstimatorPandas)
    InferentialStatisticsFactory.register("chi_square_calculator", "pandas", pd_impl.ChiSquareCalculatorPandas)
    InferentialStatisticsFactory.register("confidence_interval_calculator", "pandas", pd_impl.ConfidenceIntervalCalculatorPandas)
    InferentialStatisticsFactory.register("correlation_significance_calculator", "pandas", pd_impl.CorrelationSignificanceCalculatorPandas)
    InferentialStatisticsFactory.register("effect_size_calculator", "pandas", pd_impl.EffectSizeCalculatorPandas)
    InferentialStatisticsFactory.register("hypothesis_test_suite", "pandas", pd_impl.HypothesisTestSuitePandas)
    InferentialStatisticsFactory.register("power_analysis_calculator", "pandas", pd_impl.PowerAnalysisCalculatorPandas)


_register()
