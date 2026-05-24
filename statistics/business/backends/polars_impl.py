"""Polars statistics backends — `business`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.business.backends import pandas_impl
from statistics.business.backends.pandas_impl import *

from statistics.business.abstract import *

class PeriodChurnCalculatorPolars(AbstractPeriodChurnCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PeriodChurnCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ChurnFromEventsCalculatorPolars(AbstractChurnFromEventsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ChurnFromEventsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ChurnRateCalculatorPolars(AbstractChurnRateCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ChurnRateCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class FunnelMetricsComputerPolars(AbstractFunnelMetricsComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FunnelMetricsComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class FunnelFromEventsBuilderPolars(AbstractFunnelFromEventsBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FunnelFromEventsBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class ConversionFunnelCalculatorPolars(AbstractConversionFunnelCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ConversionFunnelCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SimpleCLVCalculatorPolars(AbstractSimpleCLVCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SimpleCLVCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DiscountedCLVCalculatorPolars(AbstractDiscountedCLVCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DiscountedCLVCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CLVSegmentAssignerPolars(AbstractCLVSegmentAssigner[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CLVSegmentAssignerPandas()

    def assign(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class CustomerLifetimeValueCalculatorPolars(AbstractCustomerLifetimeValueCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CustomerLifetimeValueCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseRatioCalculatorPolars(AbstractBaseRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseRatioCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class GrossMarginCalculatorPolars(AbstractGrossMarginCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GrossMarginCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class NetMarginCalculatorPolars(AbstractNetMarginCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NetMarginCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class ROECalculatorPolars(AbstractROECalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ROECalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class ROACalculatorPolars(AbstractROACalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ROACalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class CurrentRatioCalculatorPolars(AbstractCurrentRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CurrentRatioCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class QuickRatioCalculatorPolars(AbstractQuickRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = QuickRatioCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class DebtToEquityCalculatorPolars(AbstractDebtToEquityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DebtToEquityCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class AssetTurnoverCalculatorPolars(AbstractAssetTurnoverCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AssetTurnoverCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class InventoryTurnoverCalculatorPolars(AbstractInventoryTurnoverCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = InventoryTurnoverCalculatorPandas()

    def name(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class FinancialRatiosCalculatorPolars(AbstractFinancialRatiosCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = FinancialRatiosCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PeriodOverPeriodCalculatorPolars(AbstractPeriodOverPeriodCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PeriodOverPeriodCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CAGRCalculatorPolars(AbstractCAGRCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CAGRCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RollingGrowthCalculatorPolars(AbstractRollingGrowthCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingGrowthCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GrowthRatesCalculatorPolars(AbstractGrowthRatesCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GrowthRatesCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ParetoThresholdFinderPolars(AbstractParetoThresholdFinder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ParetoThresholdFinderPandas()

    def find(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.find(data, column, **kwargs)

class ParetoConcentrationCalculatorPolars(AbstractParetoConcentrationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ParetoConcentrationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ParetoAnalysisCalculatorPolars(AbstractParetoAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ParetoAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ReturnsComputerPolars(AbstractReturnsComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ReturnsComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class VaRCalculatorPolars(AbstractVaRCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = VaRCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CVaRCalculatorPolars(AbstractCVaRCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CVaRCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SharpeRatioCalculatorPolars(AbstractSharpeRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SharpeRatioCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SortinoRatioCalculatorPolars(AbstractSortinoRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SortinoRatioCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MaxDrawdownCalculatorPolars(AbstractMaxDrawdownCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MaxDrawdownCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CalmarRatioCalculatorPolars(AbstractCalmarRatioCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CalmarRatioCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RiskMetricsCalculatorPolars(AbstractRiskMetricsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RiskMetricsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SimpleRunRateProjectorPolars(AbstractSimpleRunRateProjector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SimpleRunRateProjectorPandas()

    def project(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.project(data, column, **kwargs)

class TrailingAverageProjectorPolars(AbstractTrailingAverageProjector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TrailingAverageProjectorPandas()

    def project(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.project(data, column, **kwargs)

class WeightedRecentProjectorPolars(AbstractWeightedRecentProjector[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = WeightedRecentProjectorPandas()

    def project(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.project(data, column, **kwargs)

class RunRateCalculatorPolars(AbstractRunRateCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = RunRateCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
