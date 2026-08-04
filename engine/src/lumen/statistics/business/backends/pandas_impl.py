"""Pandas adapter backend for the business statistics domain."""
from __future__ import annotations

from typing import Any
import pandas as pd

from business.abstract.churn_rate import AbstractChurnRateCalculator
from business.abstract.conversion_funnel import AbstractConversionFunnelCalculator
from business.abstract.customer_lifetime_value import AbstractCustomerLifetimeValueCalculator
from business.abstract.financial_ratios import AbstractFinancialRatiosCalculator
from business.abstract.growth_rates import AbstractGrowthRatesCalculator
from business.abstract.pareto_analysis import AbstractParetoAnalysisCalculator
from business.abstract.risk_metrics import AbstractRiskMetricsCalculator
from business.abstract.run_rate import AbstractRunRateCalculator

from business.churn_rate import ChurnRateCalculator
from business.conversion_funnel import ConversionFunnelCalculator
from business.customer_lifetime_value import CustomerLifetimeValueCalculator
from business.financial_ratios import FinancialRatiosCalculator
from business.growth_rates import GrowthRatesCalculator
from business.pareto_analysis import ParetoAnalysisCalculator
from business.risk_metrics import RiskMetricsCalculator
from business.run_rate import RunRateCalculator


class ChurnRateCalculatorPandas(AbstractChurnRateCalculator):
    def __init__(self) -> None:
        self._impl = ChurnRateCalculator()

    def calculate(
        self,
        data: Any,
        customer_id_column: str,
        start_date_column: str,
        end_date_column: str,
        analysis_start: str,
        analysis_end: str,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(
            df,
            customer_id_column,
            start_date_column,
            end_date_column,
            analysis_start,
            analysis_end
        )


class ConversionFunnelCalculatorPandas(AbstractConversionFunnelCalculator):
    def __init__(self) -> None:
        self._impl = ConversionFunnelCalculator()

    def calculate(
        self,
        data: Any,
        step_column: str,
        user_column: str,
        steps_order: list[str],
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, step_column, user_column, steps_order)


class CustomerLifetimeValueCalculatorPandas(AbstractCustomerLifetimeValueCalculator):
    def __init__(self) -> None:
        self._impl = CustomerLifetimeValueCalculator()

    def calculate(
        self,
        data: Any,
        customer_column: str,
        order_value_column: str,
        date_column: str,
        discount_rate: float = 0.1,
        margin_rate: float = 0.3,
        periods_per_year: int = 12,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(
            df,
            customer_column,
            order_value_column,
            date_column,
            discount_rate,
            margin_rate,
            periods_per_year
        )


class FinancialRatiosCalculatorPandas(AbstractFinancialRatiosCalculator):
    def __init__(self) -> None:
        self._impl = FinancialRatiosCalculator()

    def calculate(
        self,
        data: Any,
        revenue_column: str,
        cost_column: str,
        equity_column: str | None = None,
        assets_column: str | None = None,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, revenue_column, cost_column, equity_column, assets_column)


class GrowthRatesCalculatorPandas(AbstractGrowthRatesCalculator):
    def __init__(self) -> None:
        self._impl = GrowthRatesCalculator()

    def calculate(
        self,
        data: Any,
        date_column: str,
        value_column: str,
        periods: int = 1,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, date_column, value_column, periods)


class ParetoAnalysisCalculatorPandas(AbstractParetoAnalysisCalculator):
    def __init__(self) -> None:
        self._impl = ParetoAnalysisCalculator()

    def calculate(
        self,
        data: Any,
        entity_column: str,
        value_column: str,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, entity_column, value_column)


class RiskMetricsCalculatorPandas(AbstractRiskMetricsCalculator):
    def __init__(self) -> None:
        self._impl = RiskMetricsCalculator()

    def calculate(
        self,
        data: Any,
        returns_column: str,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, returns_column, risk_free_rate, confidence_level)


class RunRateCalculatorPandas(AbstractRunRateCalculator):
    def __init__(self) -> None:
        self._impl = RunRateCalculator()

    def calculate(
        self,
        data: Any,
        date_column: str,
        value_column: str,
        extrapolation_periods: int = 12,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, date_column, value_column, extrapolation_periods)
