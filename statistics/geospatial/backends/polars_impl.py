"""Polars statistics backends — `geospatial`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.geospatial.backends import pandas_impl
from statistics.geospatial.backends.pandas_impl import *

from statistics.geospatial.abstract import *

class CentroidCalculatorPolars(AbstractCentroidCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CentroidCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BoundingBoxCalculatorPolars(AbstractBoundingBoxCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BoundingBoxCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DiagonalDistanceCalculatorPolars(AbstractDiagonalDistanceCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DiagonalDistanceCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MeanRadiusCalculatorPolars(AbstractMeanRadiusCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanRadiusCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DispersionLabelAssignerPolars(AbstractDispersionLabelAssigner[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DispersionLabelAssignerPandas()

    def assign(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class GeoBoundingBoxCalculatorPolars(AbstractGeoBoundingBoxCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoBoundingBoxCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class HaversineDistanceMatrixPolars(AbstractHaversineDistanceMatrix[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HaversineDistanceMatrixPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class GeoClusterProfileBuilderPolars(AbstractGeoClusterProfileBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoClusterProfileBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class GeoClusteringCalculatorPolars(AbstractGeoClusteringCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoClusteringCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoUnitFrequencyCalculatorPolars(AbstractGeoUnitFrequencyCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoUnitFrequencyCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoConcentrationCalculatorPolars(AbstractGeoConcentrationCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoConcentrationCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoDistributionCalculatorPolars(AbstractGeoDistributionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoDistributionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GridBoundaryComputerPolars(AbstractGridBoundaryComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GridBoundaryComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class GridCountAccumulatorPolars(AbstractGridCountAccumulator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GridCountAccumulatorPandas()

    def accumulate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.accumulate(data, column, **kwargs)

class CellDensityCalculatorPolars(AbstractCellDensityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CellDensityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoHeatmapCalculatorPolars(AbstractGeoHeatmapCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoHeatmapCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class HaversineVectorizedCalculatorPolars(AbstractHaversineVectorizedCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = HaversineVectorizedCalculatorPandas()

    def distances_from_point(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.distances_from_point(data, column, **kwargs)

class NearestNeighborFinderPolars(AbstractNearestNeighborFinder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NearestNeighborFinderPandas()

    def find_all(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.find_all(data, column, **kwargs)

class AverageNearestNeighborIndexCalculatorPolars(AbstractAverageNearestNeighborIndexCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AverageNearestNeighborIndexCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ProximityAnalysisCalculatorPolars(AbstractProximityAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ProximityAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
