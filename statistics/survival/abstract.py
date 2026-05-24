"""Abstract statistics contracts — domain `survival`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractIntervalExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRegularityClassifier(ABC, Generic[T]):

    @abstractmethod
    def classify(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTemporalBurstinessCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRollingEventRateCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEventDensityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNelsonAalenEstimator(ABC, Generic[T]):

    @abstractmethod
    def estimate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractKernelHazardSmoother(ABC, Generic[T]):

    @abstractmethod
    def smooth(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHazardRateCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractKaplanMeierEstimator(ABC, Generic[T]):

    @abstractmethod
    def estimate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMedianSurvivalEstimator(ABC, Generic[T]):

    @abstractmethod
    def estimate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractKaplanMeierCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTimeToEventSummaryCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractThresholdSurvivalAnalyser(ABC, Generic[T]):

    @abstractmethod
    def analyse(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractExponentialFitter(ABC, Generic[T]):

    @abstractmethod
    def fit(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTimeToEventCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
