"""Polars statistics backends — `graphs`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.graphs.backends import pandas_impl
from statistics.graphs.backends.pandas_impl import *

from statistics.graphs.abstract import *

class DegreeCentralityCalculatorPolars(AbstractDegreeCentralityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DegreeCentralityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BFSShortestPathsPolars(AbstractBFSShortestPaths[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BFSShortestPathsPandas()

    def from_source(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.from_source(data, column, **kwargs)

class BetweennessCalculatorPolars(AbstractBetweennessCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BetweennessCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ClosenessCentralityCalculatorPolars(AbstractClosenessCentralityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ClosenessCentralityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PageRankCalculatorPolars(AbstractPageRankCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PageRankCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CentralityRankerPolars(AbstractCentralityRanker[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralityRankerPandas()

    def rank(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.rank(data, column, **kwargs)

class CentralityAnalysisCalculatorPolars(AbstractCentralityAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralityAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ModularityCalculatorPolars(AbstractModularityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ModularityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GreedyModularityOptimizerPolars(AbstractGreedyModularityOptimizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = GreedyModularityOptimizerPandas()

    def optimize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.optimize(data, column, **kwargs)

class CommunityProfileBuilderPolars(AbstractCommunityProfileBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CommunityProfileBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class CommunityDetectionCalculatorPolars(AbstractCommunityDetectionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CommunityDetectionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AdjacencyMatrixBuilderPolars(AbstractAdjacencyMatrixBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AdjacencyMatrixBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class ConnectedComponentsFinderPolars(AbstractConnectedComponentsFinder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ConnectedComponentsFinderPandas()

    def find(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.find(data, column, **kwargs)

class DegreeDistributionCalculatorPolars(AbstractDegreeDistributionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DegreeDistributionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class NetworkDensityCalculatorPolars(AbstractNetworkDensityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NetworkDensityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AllPairsShortestPathCalculatorPolars(AbstractAllPairsShortestPathCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = AllPairsShortestPathCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PathStatisticsExtractorPolars(AbstractPathStatisticsExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PathStatisticsExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class SmallWorldCoefficientPolars(AbstractSmallWorldCoefficient[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SmallWorldCoefficientPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ClusteringCoefficientCalculatorPolars(AbstractClusteringCoefficientCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = ClusteringCoefficientCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PathAnalysisCalculatorPolars(AbstractPathAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PathAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
