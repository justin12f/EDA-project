"""Spark statistics backends — `geospatial`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.geospatial.abstract import *

from statistics.geospatial.backends import pandas_impl
from statistics.geospatial.backends.pandas_impl import *

class CentroidCalculatorSpark(AbstractCentroidCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CentroidCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BoundingBoxCalculatorSpark(AbstractBoundingBoxCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BoundingBoxCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DiagonalDistanceCalculatorSpark(AbstractDiagonalDistanceCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DiagonalDistanceCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class MeanRadiusCalculatorSpark(AbstractMeanRadiusCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = MeanRadiusCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DispersionLabelAssignerSpark(AbstractDispersionLabelAssigner[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DispersionLabelAssignerPandas()

    def assign(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class GeoBoundingBoxCalculatorSpark(AbstractGeoBoundingBoxCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoBoundingBoxCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class HaversineDistanceMatrixSpark(AbstractHaversineDistanceMatrix[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HaversineDistanceMatrixPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class GeoClusterProfileBuilderSpark(AbstractGeoClusterProfileBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoClusterProfileBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class GeoClusteringCalculatorSpark(AbstractGeoClusteringCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoClusteringCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoUnitFrequencyCalculatorSpark(AbstractGeoUnitFrequencyCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoUnitFrequencyCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoConcentrationCalculatorSpark(AbstractGeoConcentrationCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoConcentrationCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoDistributionCalculatorSpark(AbstractGeoDistributionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoDistributionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GridBoundaryComputerSpark(AbstractGridBoundaryComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GridBoundaryComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class GridCountAccumulatorSpark(AbstractGridCountAccumulator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GridCountAccumulatorPandas()

    def accumulate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.accumulate(data, column, **kwargs)

class CellDensityCalculatorSpark(AbstractCellDensityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CellDensityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GeoHeatmapCalculatorSpark(AbstractGeoHeatmapCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GeoHeatmapCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class HaversineVectorizedCalculatorSpark(AbstractHaversineVectorizedCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = HaversineVectorizedCalculatorPandas()

    def distances_from_point(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.distances_from_point(data, column, **kwargs)

class NearestNeighborFinderSpark(AbstractNearestNeighborFinder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NearestNeighborFinderPandas()

    def find_all(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.find_all(data, column, **kwargs)

class AverageNearestNeighborIndexCalculatorSpark(AbstractAverageNearestNeighborIndexCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AverageNearestNeighborIndexCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ProximityAnalysisCalculatorSpark(AbstractProximityAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ProximityAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
