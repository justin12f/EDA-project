"""Factory for the inferential statistics domain."""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class InferentialStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the inferential statistics domain."""


def _register() -> None:
    from lumen.statistics.inferential.backends import polars_impl as pl_impl
    try:
        from lumen.statistics.inferential.backends import spark_impl as sp_impl
    except ImportError:  # PySpark is the optional `spark` extra
        sp_impl = None
    from lumen.statistics.inferential.backends import pandas_impl as pd_impl

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
    if sp_impl is not None:
        InferentialStatisticsFactory.register("anova_calculator", "spark", sp_impl.ANOVACalculatorSpark)
    if sp_impl is not None:
        InferentialStatisticsFactory.register("bootstrap_estimator", "spark", sp_impl.BootstrapEstimatorSpark)
    if sp_impl is not None:
        InferentialStatisticsFactory.register("chi_square_calculator", "spark", sp_impl.ChiSquareCalculatorSpark)
    if sp_impl is not None:
        InferentialStatisticsFactory.register("confidence_interval_calculator", "spark", sp_impl.ConfidenceIntervalCalculatorSpark)
    if sp_impl is not None:
        InferentialStatisticsFactory.register("correlation_significance_calculator", "spark", sp_impl.CorrelationSignificanceCalculatorSpark)
    if sp_impl is not None:
        InferentialStatisticsFactory.register("effect_size_calculator", "spark", sp_impl.EffectSizeCalculatorSpark)
    if sp_impl is not None:
        InferentialStatisticsFactory.register("hypothesis_test_suite", "spark", sp_impl.HypothesisTestSuiteSpark)
    if sp_impl is not None:
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
