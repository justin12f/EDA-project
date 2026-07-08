"""Dependency injection container for the business statistics domain."""
from __future__ import annotations

from typing import Literal

from business.abstract.churn_rate import AbstractChurnRateCalculator
from business.abstract.conversion_funnel import AbstractConversionFunnelCalculator
from business.abstract.customer_lifetime_value import AbstractCustomerLifetimeValueCalculator
from business.abstract.financial_ratios import AbstractFinancialRatiosCalculator
from business.abstract.growth_rates import AbstractGrowthRatesCalculator
from business.abstract.pareto_analysis import AbstractParetoAnalysisCalculator
from business.abstract.risk_metrics import AbstractRiskMetricsCalculator
from business.abstract.run_rate import AbstractRunRateCalculator
from business.factory import BusinessStatisticsFactory

Backend = Literal["polars", "spark", "pandas"]


class BusinessDependencyContainer:
    def __init__(self, backend: Backend) -> None:
        self._backend: Backend = backend
        self._factory = BusinessStatisticsFactory

        self._churn_rate: AbstractChurnRateCalculator | None = None
        self._conversion_funnel: AbstractConversionFunnelCalculator | None = None
        self._customer_lifetime_value: AbstractCustomerLifetimeValueCalculator | None = None
        self._financial_ratios: AbstractFinancialRatiosCalculator | None = None
        self._growth_rates: AbstractGrowthRatesCalculator | None = None
        self._pareto_analysis: AbstractParetoAnalysisCalculator | None = None
        self._risk_metrics: AbstractRiskMetricsCalculator | None = None
        self._run_rate: AbstractRunRateCalculator | None = None

    @property
    def backend(self) -> Backend:
        return self._backend

    def churn_rate_calculator(self) -> AbstractChurnRateCalculator:
        if self._churn_rate is None:
            self._churn_rate = self._factory.create("churn_rate_calculator", self._backend)
        return self._churn_rate

    def conversion_funnel_calculator(self) -> AbstractConversionFunnelCalculator:
        if self._conversion_funnel is None:
            self._conversion_funnel = self._factory.create("conversion_funnel_calculator", self._backend)
        return self._conversion_funnel

    def customer_lifetime_value_calculator(self) -> AbstractCustomerLifetimeValueCalculator:
        if self._customer_lifetime_value is None:
            self._customer_lifetime_value = self._factory.create("customer_lifetime_value_calculator", self._backend)
        return self._customer_lifetime_value

    def financial_ratios_calculator(self) -> AbstractFinancialRatiosCalculator:
        if self._financial_ratios is None:
            self._financial_ratios = self._factory.create("financial_ratios_calculator", self._backend)
        return self._financial_ratios

    def growth_rates_calculator(self) -> AbstractGrowthRatesCalculator:
        if self._growth_rates is None:
            self._growth_rates = self._factory.create("growth_rates_calculator", self._backend)
        return self._growth_rates

    def pareto_analysis_calculator(self) -> AbstractParetoAnalysisCalculator:
        if self._pareto_analysis is None:
            self._pareto_analysis = self._factory.create("pareto_analysis_calculator", self._backend)
        return self._pareto_analysis

    def risk_metrics_calculator(self) -> AbstractRiskMetricsCalculator:
        if self._risk_metrics is None:
            self._risk_metrics = self._factory.create("risk_metrics_calculator", self._backend)
        return self._risk_metrics

    def run_rate_calculator(self) -> AbstractRunRateCalculator:
        if self._run_rate is None:
            self._run_rate = self._factory.create("run_rate_calculator", self._backend)
        return self._run_rate
