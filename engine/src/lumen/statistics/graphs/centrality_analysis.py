"""Node centrality: degree, betweenness, closeness, and PageRank."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `GraphStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class NodeCentralityRecord:
    """Immutable centrality record for a single node."""

    node: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float
    overall_rank: int

class DegreeCentralityCalculator:
    """Degree centrality: fraction of possible connections a node has.

    DC(v) = deg(v) / (n - 1)

    Normalized by (n-1) so the maximum possible is 1.0.
    For directed graphs, uses total degree (in + out).
    """

    def calculate(
        self, adjacency: np.ndarray
    ) -> np.ndarray:
        """Compute normalized degree centrality for all nodes.

        Args:
            adjacency: Square adjacency matrix.

        Returns:
            Degree centrality array (one value per node).
        """
        n = adjacency.shape[0]
        sym = (adjacency + adjacency.T) > 0
        degrees = sym.sum(axis=1).astype(float)
        return degrees / (n - 1) if n > 1 else degrees

class BFSShortestPaths:
    """Computes shortest path lengths from a single source via BFS.

    Works on unweighted binary adjacency (ignores weights).
    Returns inf for unreachable nodes.
    """

    def from_source(
        self, adjacency: np.ndarray, source: int
    ) -> np.ndarray:
        """BFS shortest path lengths from source to all nodes.

        Args:
            adjacency: Binary adjacency matrix (symmetrized).
            source: Source node index.

        Returns:
            Distance array — inf for unreachable nodes.
        """
        n = adjacency.shape[0]
        dist = np.full(n, np.inf)
        dist[source] = 0
        queue = [source]
        sym = (adjacency + adjacency.T) > 0

        while queue:
            node = queue.pop(0)
            neighbors = np.where(sym[node])[0]
            for nb in neighbors:
                if np.isinf(dist[nb]):
                    dist[nb] = dist[node] + 1
                    queue.append(int(nb))

        return dist

class BetweennessCalculator:
    """Betweenness centrality via Brandes-like BFS accumulation.

    BC(v) = Σ_{s≠v≠t} [σ(s,t|v) / σ(s,t)] / [(n-1)(n-2)/2]

    Approximated here using shortest path counting via BFS.
    Full Brandes O(n×e) — this implementation is O(n²×e) for clarity.
    """

    def __init__(self) -> None:
        self._bfs = BFSShortestPaths()

    def calculate(self, adjacency: np.ndarray) -> np.ndarray:
        """Compute normalized betweenness centrality.

        Args:
            adjacency: Square adjacency matrix.

        Returns:
            Betweenness centrality array normalized to [0, 1].
        """
        n = adjacency.shape[0]
        betweenness = np.zeros(n)
        sym = ((adjacency + adjacency.T) > 0).astype(float)

        for source in range(n):
            dist = self._bfs.from_source(sym, source)
            for target in range(n):
                if target == source or np.isinf(dist[target]):
                    continue
                # Count intermediate nodes on shortest paths
                path_length = int(dist[target])
                if path_length < 2:
                    continue
                # Simple approximation: mark intermediate nodes
                current = target
                path_dist = self._bfs.from_source(sym, target)
                for mid in range(n):
                    if mid == source or mid == target:
                        continue
                    if (
                        abs(dist[mid] + path_dist[source] - path_length) < 1e-6
                        and not np.isinf(dist[mid])
                        and not np.isinf(path_dist[source])
                    ):
                        betweenness[mid] += 1

        norm = (n - 1) * (n - 2) / 2 if n > 2 else 1.0
        return betweenness / norm if norm > 0 else betweenness

class ClosenessCentralityCalculator:
    """Closeness centrality: inverse of mean shortest path to all reachable nodes.

    CC(v) = (n_reachable - 1)² / [(n - 1) × Σ d(v, u)]

    Wasserman-Faust normalization handles disconnected graphs:
    multiplies by (n_reachable - 1) / (n - 1) to account for
    nodes unreachable from v.
    """

    def __init__(self) -> None:
        self._bfs = BFSShortestPaths()

    def calculate(self, adjacency: np.ndarray) -> np.ndarray:
        """Compute normalized closeness centrality.

        Args:
            adjacency: Square adjacency matrix.

        Returns:
            Closeness centrality array in [0, 1].
        """
        n = adjacency.shape[0]
        closeness = np.zeros(n)
        sym = ((adjacency + adjacency.T) > 0).astype(float)

        for v in range(n):
            dist = self._bfs.from_source(sym, v)
            reachable = dist[~np.isinf(dist)]
            n_reachable = len(reachable)
            total_dist = reachable.sum()

            if n_reachable <= 1 or total_dist == 0:
                closeness[v] = 0.0
            else:
                raw = (n_reachable - 1) / total_dist
                # Wasserman-Faust normalization
                closeness[v] = raw * ((n_reachable - 1) / (n - 1))

        return closeness

class PageRankCalculator:
    """PageRank via power iteration.

    PR(v) = (1-d)/n + d × Σ_{u→v} PR(u)/out_degree(u)

    d = damping factor (standard = 0.85).
    Converges when max change < tolerance.
    """

    _DEFAULT_DAMPING: float = 0.85
    _DEFAULT_MAX_ITER: int = 100
    _DEFAULT_TOLERANCE: float = 1e-6

    def calculate(
        self,
        adjacency: np.ndarray,
        damping: float = _DEFAULT_DAMPING,
        max_iterations: int = _DEFAULT_MAX_ITER,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> np.ndarray:
        """Compute PageRank via power iteration.

        Args:
            adjacency: Square adjacency matrix.
            damping: Damping factor in (0, 1).
            max_iterations: Maximum iterations.
            tolerance: Convergence threshold.

        Returns:
            PageRank array (sums to 1.0).
        """
        n = adjacency.shape[0]
        out_degree = adjacency.sum(axis=1)

        # Build column-stochastic transition matrix
        transition = np.zeros((n, n))
        for i in range(n):
            if out_degree[i] > 0:
                transition[:, i] = adjacency[i] / out_degree[i]
            else:
                # Dangling node: distribute uniformly
                transition[:, i] = 1.0 / n

        pr = np.ones(n) / n

        for _ in range(max_iterations):
            new_pr = (1.0 - damping) / n + damping * transition @ pr
            if float(np.abs(new_pr - pr).max()) < tolerance:
                pr = new_pr
                break
            pr = new_pr

        return pr

class CentralityRanker:
    """Ranks nodes by composite centrality score."""

    def rank(
        self,
        nodes: list,
        degree: np.ndarray,
        betweenness: np.ndarray,
        closeness: np.ndarray,
        pagerank: np.ndarray,
        top_n: int | None,
    ) -> list[NodeCentralityRecord]:
        """Build ranked node centrality records.

        Args:
            nodes: Ordered node label list.
            degree: Degree centrality array.
            betweenness: Betweenness centrality array.
            closeness: Closeness centrality array.
            pagerank: PageRank array.
            top_n: Limit to top N by PageRank. None = all.

        Returns:
            List of NodeCentralityRecord sorted by PageRank descending.
        """
        sorted_idx = np.argsort(pagerank)[::-1]

        records: list[NodeCentralityRecord] = [
            NodeCentralityRecord(
                node=str(nodes[i]),
                degree_centrality=round(float(degree[i]), 6),
                betweenness_centrality=round(float(betweenness[i]), 6),
                closeness_centrality=round(float(closeness[i]), 6),
                pagerank=round(float(pagerank[i]), 6),
                overall_rank=rank + 1,
            )
            for rank, i in enumerate(sorted_idx)
        ]

        return records[:top_n] if top_n is not None else records

class CentralityAnalysisCalculator:
    """Degree, betweenness, closeness, and PageRank centrality.

    Note: Betweenness is O(n²×e) — for graphs with n > 500 nodes,
    consider setting top_n to reduce output or using approximate methods.

    Workflow:
        calculator = CentralityAnalysisCalculator()
        result = calculator.calculate(
            edges=df,
            source_column="from_node",
            target_column="to_node",
            graph_type="undirected",
            top_n=20,
            damping=0.85,
        )
    """

    _MINIMUM_NODES: int = 3
    _BETWEENNESS_NODE_LIMIT: int = 300

    def __init__(self) -> None:
        self._matrix_builder = None  # imported inline to avoid circular
        self._degree_calc = DegreeCentralityCalculator()
        self._betweenness_calc = BetweennessCalculator()
        self._closeness_calc = ClosenessCentralityCalculator()
        self._pagerank_calc = PageRankCalculator()
        self._ranker = CentralityRanker()

    def calculate(
        self,
        edges: pd.DataFrame,
        source_column: str,
        target_column: str,
        graph_type: str = "undirected",
        top_n: int | None = 20,
        damping: float = 0.85,
        weight_column: str | None = None,
    ) -> dict:
        """Compute all centrality metrics for the graph.

        Args:
            edges: Edge list DataFrame.
            source_column: Source node column.
            target_column: Target node column.
            graph_type: 'directed' or 'undirected'.
            top_n: Return top N nodes by PageRank.
            damping: PageRank damping factor.
            weight_column: Optional edge weight column.

        Returns:
            Dict with ranked node centrality scores and graph summary.

        Raises:
            KeyError: If columns are not found.
            ValueError: If parameters are invalid.
        """
        from lumen.statistics.graphs.network_density import (
            AdjacencyMatrixBuilder, GraphType
        )

        _VALID_TYPES: frozenset[str] = frozenset({"directed", "undirected"})

        for col in (source_column, target_column):
            if col not in edges.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if graph_type not in _VALID_TYPES:
            raise ValueError(
                f"graph_type must be one of {_VALID_TYPES}. Got '{graph_type}'."
            )
        if not 0.0 < damping < 1.0:
            raise ValueError(f"damping must be in (0, 1). Got {damping}.")

        gtype = GraphType(graph_type)
        adjacency, nodes = AdjacencyMatrixBuilder().build(
            edges, source_column, target_column, gtype, weight_column
        )

        if len(nodes) < self._MINIMUM_NODES:
            raise ValueError(
                f"At least {self._MINIMUM_NODES} nodes required. Got {len(nodes)}."
            )

        degree = self._degree_calc.calculate(adjacency)
        closeness = self._closeness_calc.calculate(adjacency)
        pagerank = self._pagerank_calc.calculate(adjacency, damping)

        skip_betweenness = len(nodes) > self._BETWEENNESS_NODE_LIMIT
        betweenness = (
            np.zeros(len(nodes))
            if skip_betweenness
            else self._betweenness_calc.calculate(adjacency)
        )

        ranked = self._ranker.rank(nodes, degree, betweenness, closeness, pagerank, top_n)

        return {
            "nodes": [
                {
                    "node": r.node,
                    "degree_centrality": r.degree_centrality,
                    "betweenness_centrality": r.betweenness_centrality,
                    "closeness_centrality": r.closeness_centrality,
                    "pagerank": r.pagerank,
                    "overall_rank": r.overall_rank,
                }
                for r in ranked
            ],
            "top_node_by_pagerank": ranked[0].node if ranked else None,
            "top_node_by_degree": max(ranked, key=lambda r: r.degree_centrality).node if ranked else None,
            "betweenness_skipped": skip_betweenness,
            "betweenness_skip_reason": (
                f"Graph has {len(nodes)} nodes > limit of {self._BETWEENNESS_NODE_LIMIT}."
                if skip_betweenness else None
            ),
            "n_nodes_total": len(nodes),
            "top_n": top_n,
            "damping": damping,
        }