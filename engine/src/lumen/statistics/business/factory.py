"""Factory for the business statistics domain."""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class BusinessStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the business statistics domain."""


def _register() -> None:
    from business.backends import polars_impl as pl_impl
    from business.backends import spark_impl as sp_impl
    from business.backends import pandas_impl as pd_impl

    # --- Polars ---
    BusinessStatisticsFactory.register("churn_rate_calculator", "polars", pl_impl.ChurnRateCalculatorPolars)
    BusinessStatisticsFactory.register("conversion_funnel_calculator", "polars", pl_impl.ConversionFunnelCalculatorPolars)
    BusinessStatisticsFactory.register("customer_lifetime_value_calculator", "polars", pl_impl.CustomerLifetimeValueCalculatorPolars)
    BusinessStatisticsFactory.register("financial_ratios_calculator", "polars", pl_impl.FinancialRatiosCalculatorPolars)
    BusinessStatisticsFactory.register("growth_rates_calculator", "polars", pl_impl.GrowthRatesCalculatorPolars)
    BusinessStatisticsFactory.register("pareto_analysis_calculator", "polars", pl_impl.ParetoAnalysisCalculatorPolars)
    BusinessStatisticsFactory.register("risk_metrics_calculator", "polars", pl_impl.RiskMetricsCalculatorPolars)
    BusinessStatisticsFactory.register("run_rate_calculator", "polars", pl_impl.RunRateCalculatorPolars)

    # --- Spark ---
    BusinessStatisticsFactory.register("churn_rate_calculator", "spark", sp_impl.ChurnRateCalculatorSpark)
    BusinessStatisticsFactory.register("conversion_funnel_calculator", "spark", sp_impl.ConversionFunnelCalculatorSpark)
    BusinessStatisticsFactory.register("customer_lifetime_value_calculator", "spark", sp_impl.CustomerLifetimeValueCalculatorSpark)
    BusinessStatisticsFactory.register("financial_ratios_calculator", "spark", sp_impl.FinancialRatiosCalculatorSpark)
    BusinessStatisticsFactory.register("growth_rates_calculator", "spark", sp_impl.GrowthRatesCalculatorSpark)
    BusinessStatisticsFactory.register("pareto_analysis_calculator", "spark", sp_impl.ParetoAnalysisCalculatorSpark)
    BusinessStatisticsFactory.register("risk_metrics_calculator", "spark", sp_impl.RiskMetricsCalculatorSpark)
    BusinessStatisticsFactory.register("run_rate_calculator", "spark", sp_impl.RunRateCalculatorSpark)

    # --- Pandas ---
    BusinessStatisticsFactory.register("churn_rate_calculator", "pandas", pd_impl.ChurnRateCalculatorPandas)
    BusinessStatisticsFactory.register("conversion_funnel_calculator", "pandas", pd_impl.ConversionFunnelCalculatorPandas)
    BusinessStatisticsFactory.register("customer_lifetime_value_calculator", "pandas", pd_impl.CustomerLifetimeValueCalculatorPandas)
    BusinessStatisticsFactory.register("financial_ratios_calculator", "pandas", pd_impl.FinancialRatiosCalculatorPandas)
    BusinessStatisticsFactory.register("growth_rates_calculator", "pandas", pd_impl.GrowthRatesCalculatorPandas)
    BusinessStatisticsFactory.register("pareto_analysis_calculator", "pandas", pd_impl.ParetoAnalysisCalculatorPandas)
    BusinessStatisticsFactory.register("risk_metrics_calculator", "pandas", pd_impl.RiskMetricsCalculatorPandas)
    BusinessStatisticsFactory.register("run_rate_calculator", "pandas", pd_impl.RunRateCalculatorPandas)


_register()
