"""Pandas statistics backends — `graphs`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.graphs.abstract import *

import statistics.graphs.centrality_analysis as _mod_centrality_analysis
import statistics.graphs.community_detection as _mod_community_detection
import statistics.graphs.network_density as _mod_network_density
import statistics.graphs.path_analysis as _mod_path_analysis

class DegreeCentralityCalculatorPandas(AbstractDegreeCentralityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_centrality_analysis.DegreeCentralityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BFSShortestPathsPandas(AbstractBFSShortestPaths[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_centrality_analysis.BFSShortestPaths()

    def from_source(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.from_source(arr, **kwargs)

class BetweennessCalculatorPandas(AbstractBetweennessCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_centrality_analysis.BetweennessCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ClosenessCentralityCalculatorPandas(AbstractClosenessCentralityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_centrality_analysis.ClosenessCentralityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class PageRankCalculatorPandas(AbstractPageRankCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_centrality_analysis.PageRankCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class CentralityRankerPandas(AbstractCentralityRanker[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_centrality_analysis.CentralityRanker()

    def rank(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.rank(arr, **kwargs)

class CentralityAnalysisCalculatorPandas(AbstractCentralityAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_centrality_analysis.CentralityAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ModularityCalculatorPandas(AbstractModularityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_community_detection.ModularityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class GreedyModularityOptimizerPandas(AbstractGreedyModularityOptimizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_community_detection.GreedyModularityOptimizer()

    def optimize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.optimize(arr, **kwargs)

class CommunityProfileBuilderPandas(AbstractCommunityProfileBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_community_detection.CommunityProfileBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class CommunityDetectionCalculatorPandas(AbstractCommunityDetectionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_community_detection.CommunityDetectionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class AdjacencyMatrixBuilderPandas(AbstractAdjacencyMatrixBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_network_density.AdjacencyMatrixBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class ConnectedComponentsFinderPandas(AbstractConnectedComponentsFinder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_network_density.ConnectedComponentsFinder()

    def find(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.find(arr, **kwargs)

class DegreeDistributionCalculatorPandas(AbstractDegreeDistributionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_network_density.DegreeDistributionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class NetworkDensityCalculatorPandas(AbstractNetworkDensityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_network_density.NetworkDensityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class AllPairsShortestPathCalculatorPandas(AbstractAllPairsShortestPathCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_path_analysis.AllPairsShortestPathCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class PathStatisticsExtractorPandas(AbstractPathStatisticsExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_path_analysis.PathStatisticsExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class SmallWorldCoefficientPandas(AbstractSmallWorldCoefficient[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_path_analysis.SmallWorldCoefficient()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class ClusteringCoefficientCalculatorPandas(AbstractClusteringCoefficientCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_path_analysis.ClusteringCoefficientCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class PathAnalysisCalculatorPandas(AbstractPathAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_path_analysis.PathAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
