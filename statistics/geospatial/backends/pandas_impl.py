"""Pandas statistics backends — `geospatial`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.geospatial.abstract import *

import statistics.geospatial.geo_bounding_box as _mod_geo_bounding_box
import statistics.geospatial.geo_clustering as _mod_geo_clustering
import statistics.geospatial.geo_distribution as _mod_geo_distribution
import statistics.geospatial.geo_heatmap as _mod_geo_heatmap
import statistics.geospatial.proximity_analysis as _mod_proximity_analysis

class CentroidCalculatorPandas(AbstractCentroidCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_bounding_box.CentroidCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BoundingBoxCalculatorPandas(AbstractBoundingBoxCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_bounding_box.BoundingBoxCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class DiagonalDistanceCalculatorPandas(AbstractDiagonalDistanceCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_bounding_box.DiagonalDistanceCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class MeanRadiusCalculatorPandas(AbstractMeanRadiusCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_bounding_box.MeanRadiusCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class DispersionLabelAssignerPandas(AbstractDispersionLabelAssigner[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_bounding_box.DispersionLabelAssigner()

    def assign(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.assign(arr, **kwargs)

class GeoBoundingBoxCalculatorPandas(AbstractGeoBoundingBoxCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_bounding_box.GeoBoundingBoxCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class HaversineDistanceMatrixPandas(AbstractHaversineDistanceMatrix[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_clustering.HaversineDistanceMatrix()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class GeoClusterProfileBuilderPandas(AbstractGeoClusterProfileBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_clustering.GeoClusterProfileBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class GeoClusteringCalculatorPandas(AbstractGeoClusteringCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_clustering.GeoClusteringCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GeoUnitFrequencyCalculatorPandas(AbstractGeoUnitFrequencyCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_distribution.GeoUnitFrequencyCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GeoConcentrationCalculatorPandas(AbstractGeoConcentrationCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_distribution.GeoConcentrationCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GeoDistributionCalculatorPandas(AbstractGeoDistributionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_distribution.GeoDistributionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GridBoundaryComputerPandas(AbstractGridBoundaryComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_heatmap.GridBoundaryComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class GridCountAccumulatorPandas(AbstractGridCountAccumulator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_heatmap.GridCountAccumulator()

    def accumulate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.accumulate(arr, **kwargs)

class CellDensityCalculatorPandas(AbstractCellDensityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_heatmap.CellDensityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GeoHeatmapCalculatorPandas(AbstractGeoHeatmapCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_geo_heatmap.GeoHeatmapCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class HaversineVectorizedCalculatorPandas(AbstractHaversineVectorizedCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_proximity_analysis.HaversineVectorizedCalculator()

    def distances_from_point(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.distances_from_point(arr, **kwargs)

class NearestNeighborFinderPandas(AbstractNearestNeighborFinder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_proximity_analysis.NearestNeighborFinder()

    def find_all(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.find_all(arr, **kwargs)

class AverageNearestNeighborIndexCalculatorPandas(AbstractAverageNearestNeighborIndexCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_proximity_analysis.AverageNearestNeighborIndexCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ProximityAnalysisCalculatorPandas(AbstractProximityAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_proximity_analysis.ProximityAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
