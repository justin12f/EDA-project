"""Abstract statistics contracts — domain `graphs`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractDegreeCentralityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBFSShortestPaths(ABC, Generic[T]):

    @abstractmethod
    def from_source(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBetweennessCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractClosenessCentralityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPageRankCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCentralityRanker(ABC, Generic[T]):

    @abstractmethod
    def rank(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCentralityAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractModularityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGreedyModularityOptimizer(ABC, Generic[T]):

    @abstractmethod
    def optimize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCommunityProfileBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCommunityDetectionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAdjacencyMatrixBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractConnectedComponentsFinder(ABC, Generic[T]):

    @abstractmethod
    def find(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDegreeDistributionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNetworkDensityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAllPairsShortestPathCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPathStatisticsExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSmallWorldCoefficient(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractClusteringCoefficientCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPathAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
