"""Factory for the relational statistics domain."""
from __future__ import annotations

from lumen.core.abstract_factory import RegistryFactory


class RelationalStatisticsFactory(RegistryFactory):
    """Registry factory scoped to the relational statistics domain."""


def _register() -> None:
    from relational.backends import polars_impl as pl_impl
    from relational.backends import spark_impl as sp_impl
    from relational.backends import pandas_impl as pd_impl

    # --- Polars ---
    RelationalStatisticsFactory.register("contingency_analysis_calculator", "polars", pl_impl.ContingencyAnalysisCalculatorPolars)
    RelationalStatisticsFactory.register("correlation_matrix_calculator", "polars", pl_impl.CorrelationMatrixCalculatorPolars)
    RelationalStatisticsFactory.register("cross_correlation_calculator", "polars", pl_impl.CrossCorrelationCalculatorPolars)
    RelationalStatisticsFactory.register("granger_causality_calculator", "polars", pl_impl.GrangerCausalityCalculatorPolars)
    RelationalStatisticsFactory.register("interaction_effects_calculator", "polars", pl_impl.InteractionEffectsCalculatorPolars)
    RelationalStatisticsFactory.register("multicollinearity_calculator", "polars", pl_impl.MulticollinearityCalculatorPolars)
    RelationalStatisticsFactory.register("mutual_information_calculator", "polars", pl_impl.MutualInformationCalculatorPolars)
    RelationalStatisticsFactory.register("partial_correlation_calculator", "polars", pl_impl.PartialCorrelationCalculatorPolars)

    # --- Spark ---
    RelationalStatisticsFactory.register("contingency_analysis_calculator", "spark", sp_impl.ContingencyAnalysisCalculatorSpark)
    RelationalStatisticsFactory.register("correlation_matrix_calculator", "spark", sp_impl.CorrelationMatrixCalculatorSpark)
    RelationalStatisticsFactory.register("cross_correlation_calculator", "spark", sp_impl.CrossCorrelationCalculatorSpark)
    RelationalStatisticsFactory.register("granger_causality_calculator", "spark", sp_impl.GrangerCausalityCalculatorSpark)
    RelationalStatisticsFactory.register("interaction_effects_calculator", "spark", sp_impl.InteractionEffectsCalculatorSpark)
    RelationalStatisticsFactory.register("multicollinearity_calculator", "spark", sp_impl.MulticollinearityCalculatorSpark)
    RelationalStatisticsFactory.register("mutual_information_calculator", "spark", sp_impl.MutualInformationCalculatorSpark)
    RelationalStatisticsFactory.register("partial_correlation_calculator", "spark", sp_impl.PartialCorrelationCalculatorSpark)

    # --- Pandas ---
    RelationalStatisticsFactory.register("contingency_analysis_calculator", "pandas", pd_impl.ContingencyAnalysisCalculatorPandas)
    RelationalStatisticsFactory.register("correlation_matrix_calculator", "pandas", pd_impl.CorrelationMatrixCalculatorPandas)
    RelationalStatisticsFactory.register("cross_correlation_calculator", "pandas", pd_impl.CrossCorrelationCalculatorPandas)
    RelationalStatisticsFactory.register("granger_causality_calculator", "pandas", pd_impl.GrangerCausalityCalculatorPandas)
    RelationalStatisticsFactory.register("interaction_effects_calculator", "pandas", pd_impl.InteractionEffectsCalculatorPandas)
    RelationalStatisticsFactory.register("multicollinearity_calculator", "pandas", pd_impl.MulticollinearityCalculatorPandas)
    RelationalStatisticsFactory.register("mutual_information_calculator", "pandas", pd_impl.MutualInformationCalculatorPandas)
    RelationalStatisticsFactory.register("partial_correlation_calculator", "pandas", pd_impl.PartialCorrelationCalculatorPandas)


_register()
