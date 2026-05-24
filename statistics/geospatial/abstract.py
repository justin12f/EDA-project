"""Abstract statistics contracts — domain `geospatial`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractCentroidCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBoundingBoxCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDiagonalDistanceCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMeanRadiusCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDispersionLabelAssigner(ABC, Generic[T]):

    @abstractmethod
    def assign(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGeoBoundingBoxCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHaversineDistanceMatrix(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGeoClusterProfileBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGeoClusteringCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGeoUnitFrequencyCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGeoConcentrationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGeoDistributionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGridBoundaryComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGridCountAccumulator(ABC, Generic[T]):

    @abstractmethod
    def accumulate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCellDensityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGeoHeatmapCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHaversineVectorizedCalculator(ABC, Generic[T]):

    @abstractmethod
    def distances_from_point(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNearestNeighborFinder(ABC, Generic[T]):

    @abstractmethod
    def find_all(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAverageNearestNeighborIndexCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractProximityAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
