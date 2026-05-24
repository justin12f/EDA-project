"""Spark statistics backends — `business`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.business.abstract import *

from statistics.business.backends import pandas_impl
from statistics.business.backends.pandas_impl import *

class PeriodChurnCalculatorSpark(AbstractPeriodChurnCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PeriodChurnCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ChurnFromEventsCalculatorSpark(AbstractChurnFromEventsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ChurnFromEventsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ChurnRateCalculatorSpark(AbstractChurnRateCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ChurnRateCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class FunnelMetricsComputerSpark(AbstractFunnelMetricsComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FunnelMetricsComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class FunnelFromEventsBuilderSpark(AbstractFunnelFromEventsBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FunnelFromEventsBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class ConversionFunnelCalculatorSpark(AbstractConversionFunnelCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ConversionFunnelCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SimpleCLVCalculatorSpark(AbstractSimpleCLVCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SimpleCLVCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DiscountedCLVCalculatorSpark(AbstractDiscountedCLVCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DiscountedCLVCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CLVSegmentAssignerSpark(AbstractCLVSegmentAssigner[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CLVSegmentAssignerPandas()

    def assign(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class CustomerLifetimeValueCalculatorSpark(AbstractCustomerLifetimeValueCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CustomerLifetimeValueCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BaseRatioCalculatorSpark(AbstractBaseRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BaseRatioCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class GrossMarginCalculatorSpark(AbstractGrossMarginCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GrossMarginCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class NetMarginCalculatorSpark(AbstractNetMarginCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NetMarginCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class ROECalculatorSpark(AbstractROECalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ROECalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class ROACalculatorSpark(AbstractROACalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ROACalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class CurrentRatioCalculatorSpark(AbstractCurrentRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CurrentRatioCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class QuickRatioCalculatorSpark(AbstractQuickRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = QuickRatioCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class DebtToEquityCalculatorSpark(AbstractDebtToEquityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DebtToEquityCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class AssetTurnoverCalculatorSpark(AbstractAssetTurnoverCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AssetTurnoverCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class InventoryTurnoverCalculatorSpark(AbstractInventoryTurnoverCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = InventoryTurnoverCalculatorPandas()

    def name(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.name(data, column, **kwargs)

    def category(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.category(data, column, **kwargs)

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class FinancialRatiosCalculatorSpark(AbstractFinancialRatiosCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = FinancialRatiosCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PeriodOverPeriodCalculatorSpark(AbstractPeriodOverPeriodCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PeriodOverPeriodCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CAGRCalculatorSpark(AbstractCAGRCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CAGRCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RollingGrowthCalculatorSpark(AbstractRollingGrowthCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RollingGrowthCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GrowthRatesCalculatorSpark(AbstractGrowthRatesCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GrowthRatesCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ParetoThresholdFinderSpark(AbstractParetoThresholdFinder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ParetoThresholdFinderPandas()

    def find(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.find(data, column, **kwargs)

class ParetoConcentrationCalculatorSpark(AbstractParetoConcentrationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ParetoConcentrationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ParetoAnalysisCalculatorSpark(AbstractParetoAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ParetoAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ReturnsComputerSpark(AbstractReturnsComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ReturnsComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class VaRCalculatorSpark(AbstractVaRCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = VaRCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CVaRCalculatorSpark(AbstractCVaRCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CVaRCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SharpeRatioCalculatorSpark(AbstractSharpeRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SharpeRatioCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SortinoRatioCalculatorSpark(AbstractSortinoRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SortinoRatioCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MaxDrawdownCalculatorSpark(AbstractMaxDrawdownCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MaxDrawdownCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CalmarRatioCalculatorSpark(AbstractCalmarRatioCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CalmarRatioCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class RiskMetricsCalculatorSpark(AbstractRiskMetricsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RiskMetricsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class SimpleRunRateProjectorSpark(AbstractSimpleRunRateProjector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SimpleRunRateProjectorPandas()

    def project(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.project(data, column, **kwargs)

class TrailingAverageProjectorSpark(AbstractTrailingAverageProjector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TrailingAverageProjectorPandas()

    def project(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.project(data, column, **kwargs)

class WeightedRecentProjectorSpark(AbstractWeightedRecentProjector[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = WeightedRecentProjectorPandas()

    def project(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.project(data, column, **kwargs)

class RunRateCalculatorSpark(AbstractRunRateCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = RunRateCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
