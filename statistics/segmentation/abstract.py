"""Abstract statistics contracts — domain `segmentation`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractCohortAssigner(ABC, Generic[T]):

    @abstractmethod
    def assign(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCohortPeriodOffsetCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRetentionMatrixBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRetentionRateNormalizer(ABC, Generic[T]):

    @abstractmethod
    def normalize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCohortAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEpsilonEstimator(ABC, Generic[T]):

    @abstractmethod
    def estimate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDBSCANClusterProfileBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDBSCANClusterCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLinkageMatrixBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCopheneticCorrelationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractOptimalCutoffSelector(ABC, Generic[T]):

    @abstractmethod
    def select(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDendrogramDataExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHierarchicalClusterProfileBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHierarchicalClusterCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractElbowMethodCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSilhouetteScoreCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractOptimalKSelector(ABC, Generic[T]):

    @abstractmethod
    def select(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractClusterProfileBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractKMeansClusterCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractWelchTTestComparator(ABC, Generic[T]):

    @abstractmethod
    def compare(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCohensDComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEffectMagnitudeClassifier(ABC, Generic[T]):

    @abstractmethod
    def classify(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCategoricalDistributionComparator(ABC, Generic[T]):

    @abstractmethod
    def compare(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPopulationSplitsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRFMMetricsComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractQuantileRFMScorer(ABC, Generic[T]):

    @abstractmethod
    def score(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRFMSegmentAssigner(ABC, Generic[T]):

    @abstractmethod
    def assign(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRFMSegmentationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
