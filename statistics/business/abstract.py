"""Abstract statistics contracts — domain `business`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractPeriodChurnCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractChurnFromEventsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractChurnRateCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFunnelMetricsComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFunnelFromEventsBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractConversionFunnelCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSimpleCLVCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDiscountedCLVCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCLVSegmentAssigner(ABC, Generic[T]):

    @abstractmethod
    def assign(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCustomerLifetimeValueCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGrossMarginCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNetMarginCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractROECalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractROACalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCurrentRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractQuickRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDebtToEquityCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAssetTurnoverCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractInventoryTurnoverCalculator(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def category(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFinancialRatiosCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPeriodOverPeriodCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCAGRCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingGrowthCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGrowthRatesCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractParetoThresholdFinder(ABC, Generic[T]):

    @abstractmethod
    def find(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractParetoConcentrationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractParetoAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractReturnsComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractVaRCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCVaRCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSharpeRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSortinoRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMaxDrawdownCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCalmarRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRiskMetricsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSimpleRunRateProjector(ABC, Generic[T]):

    @abstractmethod
    def project(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTrailingAverageProjector(ABC, Generic[T]):

    @abstractmethod
    def project(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractWeightedRecentProjector(ABC, Generic[T]):

    @abstractmethod
    def project(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRunRateCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
