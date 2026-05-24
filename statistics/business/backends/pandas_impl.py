"""Pandas statistics backends — `business`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.business.abstract import *

import statistics.business.churn_rate as _mod_churn_rate
import statistics.business.conversion_funnel as _mod_conversion_funnel
import statistics.business.customer_lifetime_value as _mod_customer_lifetime_value
import statistics.business.financial_ratios as _mod_financial_ratios
import statistics.business.growth_rates as _mod_growth_rates
import statistics.business.pareto_analysis as _mod_pareto_analysis
import statistics.business.risk_metrics as _mod_risk_metrics
import statistics.business.run_rate as _mod_run_rate

class PeriodChurnCalculatorPandas(AbstractPeriodChurnCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_churn_rate.PeriodChurnCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ChurnFromEventsCalculatorPandas(AbstractChurnFromEventsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_churn_rate.ChurnFromEventsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ChurnRateCalculatorPandas(AbstractChurnRateCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_churn_rate.ChurnRateCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class FunnelMetricsComputerPandas(AbstractFunnelMetricsComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_conversion_funnel.FunnelMetricsComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class FunnelFromEventsBuilderPandas(AbstractFunnelFromEventsBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_conversion_funnel.FunnelFromEventsBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class ConversionFunnelCalculatorPandas(AbstractConversionFunnelCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_conversion_funnel.ConversionFunnelCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SimpleCLVCalculatorPandas(AbstractSimpleCLVCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_customer_lifetime_value.SimpleCLVCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class DiscountedCLVCalculatorPandas(AbstractDiscountedCLVCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_customer_lifetime_value.DiscountedCLVCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CLVSegmentAssignerPandas(AbstractCLVSegmentAssigner[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_customer_lifetime_value.CLVSegmentAssigner()

    def assign(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.assign(arr, **kwargs)

class CustomerLifetimeValueCalculatorPandas(AbstractCustomerLifetimeValueCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_customer_lifetime_value.CustomerLifetimeValueCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BaseRatioCalculatorPandas(AbstractBaseRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.BaseRatioCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class GrossMarginCalculatorPandas(AbstractGrossMarginCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.GrossMarginCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class NetMarginCalculatorPandas(AbstractNetMarginCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.NetMarginCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class ROECalculatorPandas(AbstractROECalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.ROECalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class ROACalculatorPandas(AbstractROACalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.ROACalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class CurrentRatioCalculatorPandas(AbstractCurrentRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.CurrentRatioCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class QuickRatioCalculatorPandas(AbstractQuickRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.QuickRatioCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class DebtToEquityCalculatorPandas(AbstractDebtToEquityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.DebtToEquityCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class AssetTurnoverCalculatorPandas(AbstractAssetTurnoverCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.AssetTurnoverCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class InventoryTurnoverCalculatorPandas(AbstractInventoryTurnoverCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.InventoryTurnoverCalculator()

    def name(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.name(arr, **kwargs)

    def category(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.category(arr, **kwargs)

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class FinancialRatiosCalculatorPandas(AbstractFinancialRatiosCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_financial_ratios.FinancialRatiosCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class PeriodOverPeriodCalculatorPandas(AbstractPeriodOverPeriodCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_growth_rates.PeriodOverPeriodCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CAGRCalculatorPandas(AbstractCAGRCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_growth_rates.CAGRCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RollingGrowthCalculatorPandas(AbstractRollingGrowthCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_growth_rates.RollingGrowthCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GrowthRatesCalculatorPandas(AbstractGrowthRatesCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_growth_rates.GrowthRatesCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ParetoThresholdFinderPandas(AbstractParetoThresholdFinder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_pareto_analysis.ParetoThresholdFinder()

    def find(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.find(arr, **kwargs)

class ParetoConcentrationCalculatorPandas(AbstractParetoConcentrationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_pareto_analysis.ParetoConcentrationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ParetoAnalysisCalculatorPandas(AbstractParetoAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_pareto_analysis.ParetoAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ReturnsComputerPandas(AbstractReturnsComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.ReturnsComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class VaRCalculatorPandas(AbstractVaRCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.VaRCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CVaRCalculatorPandas(AbstractCVaRCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.CVaRCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SharpeRatioCalculatorPandas(AbstractSharpeRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.SharpeRatioCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SortinoRatioCalculatorPandas(AbstractSortinoRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.SortinoRatioCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MaxDrawdownCalculatorPandas(AbstractMaxDrawdownCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.MaxDrawdownCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CalmarRatioCalculatorPandas(AbstractCalmarRatioCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.CalmarRatioCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class RiskMetricsCalculatorPandas(AbstractRiskMetricsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_risk_metrics.RiskMetricsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class SimpleRunRateProjectorPandas(AbstractSimpleRunRateProjector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_run_rate.SimpleRunRateProjector()

    def project(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.project(arr, **kwargs)

class TrailingAverageProjectorPandas(AbstractTrailingAverageProjector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_run_rate.TrailingAverageProjector()

    def project(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.project(arr, **kwargs)

class WeightedRecentProjectorPandas(AbstractWeightedRecentProjector[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_run_rate.WeightedRecentProjector()

    def project(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.project(arr, **kwargs)

class RunRateCalculatorPandas(AbstractRunRateCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_run_rate.RunRateCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
