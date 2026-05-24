"""Abstract statistics contracts — domain `time_series`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractCUSUMDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractVarianceShiftDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractChangePointDetector(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTrendRemover(ABC, Generic[T]):

    @abstractmethod
    def remove(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHanningWindowApplier(ABC, Generic[T]):

    @abstractmethod
    def apply(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFFTPowerSpectrumCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDominantCycleExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCyclicalPatternsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseAccuracyMetric(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMAEMetric(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRMSEMetric(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMAPEMetric(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMASEMetric(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractForecastAccuracyCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAutocovarianceCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractACFCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPACFCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLagFeatureBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLagFeaturesCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRateOfChangeCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAccelerationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMomentumSignalClassifier(ABC, Generic[T]):

    @abstractmethod
    def classify(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMomentumCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseMovingAverage(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSimpleMovingAverage(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractExponentialMovingAverage(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractWeightedMovingAverage(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCrossoverDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMovingAveragesCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseRollingStatistic(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingMean(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingStd(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingMin(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingMax(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingMedian(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingSkewness(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingStatisticsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCenteredMovingAverage(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEstacionalComponent(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSeasonalDecomposition(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAugmentedDickeyFullerTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractKPSSTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractStationarityVerdictInterpreter(ABC, Generic[T]):

    @abstractmethod
    def interpret(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractStationarityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingStdCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEWMAVolatilityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCoefficientOfVariationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractVolatilityRegimeDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractVolatilityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
