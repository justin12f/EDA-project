"""Spark statistics backends — `graphs`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.graphs.abstract import *

from statistics.graphs.backends import pandas_impl
from statistics.graphs.backends.pandas_impl import *

class DegreeCentralityCalculatorSpark(AbstractDegreeCentralityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DegreeCentralityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BFSShortestPathsSpark(AbstractBFSShortestPaths[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BFSShortestPathsPandas()

    def from_source(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.from_source(data, column, **kwargs)

class BetweennessCalculatorSpark(AbstractBetweennessCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BetweennessCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ClosenessCentralityCalculatorSpark(AbstractClosenessCentralityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ClosenessCentralityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PageRankCalculatorSpark(AbstractPageRankCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PageRankCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class CentralityRankerSpark(AbstractCentralityRanker[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralityRankerPandas()

    def rank(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.rank(data, column, **kwargs)

class CentralityAnalysisCalculatorSpark(AbstractCentralityAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CentralityAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ModularityCalculatorSpark(AbstractModularityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ModularityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class GreedyModularityOptimizerSpark(AbstractGreedyModularityOptimizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = GreedyModularityOptimizerPandas()

    def optimize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.optimize(data, column, **kwargs)

class CommunityProfileBuilderSpark(AbstractCommunityProfileBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CommunityProfileBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class CommunityDetectionCalculatorSpark(AbstractCommunityDetectionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CommunityDetectionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AdjacencyMatrixBuilderSpark(AbstractAdjacencyMatrixBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AdjacencyMatrixBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class ConnectedComponentsFinderSpark(AbstractConnectedComponentsFinder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ConnectedComponentsFinderPandas()

    def find(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.find(data, column, **kwargs)

class DegreeDistributionCalculatorSpark(AbstractDegreeDistributionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DegreeDistributionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class NetworkDensityCalculatorSpark(AbstractNetworkDensityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NetworkDensityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class AllPairsShortestPathCalculatorSpark(AbstractAllPairsShortestPathCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = AllPairsShortestPathCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PathStatisticsExtractorSpark(AbstractPathStatisticsExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PathStatisticsExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class SmallWorldCoefficientSpark(AbstractSmallWorldCoefficient[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SmallWorldCoefficientPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class ClusteringCoefficientCalculatorSpark(AbstractClusteringCoefficientCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = ClusteringCoefficientCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class PathAnalysisCalculatorSpark(AbstractPathAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PathAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
