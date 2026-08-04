"""Graph path analysis: average shortest path length and diameter."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `GraphStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import numpy as np
import pandas as pd

class AllPairsShortestPathCalculator:
    """Floyd-Warshall all-pairs shortest path for small graphs.

    Time complexity: O(n³). Suitable for n < 500.
    Returns inf for unreachable pairs.
    """

    def calculate(self, adjacency: np.ndarray) -> np.ndarray:
        """Compute all-pairs shortest path distance matrix.

        Args:
            adjacency: Adjacency matrix (binary or weighted).

        Returns:
            Distance matrix (n × n), inf for unreachable pairs.
        """
        n = adjacency.shape[0]
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)

        # Initialize direct edges
        connected = adjacency > 0
        dist[connected] = 1.0

        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        return dist

class PathStatisticsExtractor:
    """Extracts path statistics from an all-pairs distance matrix."""

    def extract(self, dist_matrix: np.ndarray) -> dict:
        """Compute path statistics.

        Args:
            dist_matrix: All-pairs shortest path matrix (inf for unreachable).

        Returns:
            Dict with avg_path_length, diameter, reachability_ratio, etc.
        """
        n = dist_matrix.shape[0]
        finite_distances = dist_matrix[~np.isinf(dist_matrix) & (dist_matrix > 0)]

        if len(finite_distances) == 0:
            return {
                "avg_shortest_path_length": None,
                "diameter": None,
                "radius": None,
                "reachable_pairs": 0,
                "total_pairs": n * (n - 1),
                "reachability_ratio": 0.0,
            }

        total_pairs = n * (n - 1)
        reachable = len(finite_distances)

        return {
            "avg_shortest_path_length": round(float(finite_distances.mean()), 4),
            "diameter": int(finite_distances.max()),
            "radius": int(np.min(dist_matrix.max(axis=1)[dist_matrix.max(axis=1) < np.inf])),
            "reachable_pairs": reachable,
            "total_pairs": total_pairs,
            "reachability_ratio": round(reachable / total_pairs, 4),
        }

class SmallWorldCoefficient:
    """Computes the small-world coefficient σ.

    σ = (C / C_random) / (L / L_random)

    where C = clustering coefficient, L = avg path length,
    C_random = log(n)/n, L_random = log(n)/log(log(n)).

    σ > 1 suggests small-world properties.
    """

    def calculate(
        self,
        clustering_coeff: float,
        avg_path_length: float,
        n_nodes: int,
    ) -> float | None:
        """Compute small-world coefficient.

        Args:
            clustering_coeff: Average clustering coefficient.
            avg_path_length: Average shortest path length.
            n_nodes: Number of nodes.

        Returns:
            Small-world coefficient σ or None if undefined.
        """
        if n_nodes < 3 or avg_path_length == 0:
            return None

        import math
        log_n = math.log(n_nodes)
        log_log_n = math.log(log_n) if log_n > 1 else 1.0

        c_random = log_n / n_nodes if n_nodes > 0 else 0.0
        l_random = log_n / log_log_n if log_log_n > 0 else log_n

        if c_random == 0 or l_random == 0:
            return None

        sigma = (clustering_coeff / c_random) / (avg_path_length / l_random)
        return round(float(sigma), 4)

class ClusteringCoefficientCalculator:
    """Local clustering coefficient averaged across all nodes.

    CC(v) = 2 × (triangles through v) / (deg(v) × (deg(v) - 1))

    Global = mean of local CC across all nodes with degree >= 2.
    """

    def calculate(self, adjacency: np.ndarray) -> float:
        """Compute average clustering coefficient.

        Args:
            adjacency: Binary symmetric adjacency matrix.

        Returns:
            Average clustering coefficient in [0, 1].
        """
        n = adjacency.shape[0]
        binary = (adjacency > 0).astype(float)
        np.fill_diagonal(binary, 0)
        coefficients: list[float] = []

        for v in range(n):
            neighbors = np.where(binary[v] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue
            subgraph = binary[np.ix_(neighbors, neighbors)]
            triangles = float(subgraph.sum()) / 2
            possible = k * (k - 1) / 2
            coefficients.append(triangles / possible)

        return round(float(np.mean(coefficients)), 6) if coefficients else 0.0

class PathAnalysisCalculator:
    """Average path length, diameter, clustering, and small-world coefficient.

    Note: Floyd-Warshall is O(n³) — limit to graphs with n < 500.

    Workflow:
        calculator = PathAnalysisCalculator()
        result = calculator.calculate(
            edges=df,
            source_column="from_node",
            target_column="to_node",
            graph_type="undirected",
        )
    """

    _MINIMUM_NODES: int = 3
    _NODE_LIMIT: int = 500

    def __init__(self) -> None:
        self._shortest_paths = AllPairsShortestPathCalculator()
        self._stats_extractor = PathStatisticsExtractor()
        self._clustering_calc = ClusteringCoefficientCalculator()
        self._small_world_calc = SmallWorldCoefficient()

    def calculate(
        self,
        edges: pd.DataFrame,
        source_column: str,
        target_column: str,
        graph_type: str = "undirected",
        weight_column: str | None = None,
    ) -> dict:
        """Compute path-based graph metrics.

        Args:
            edges: Edge list DataFrame.
            source_column: Source node column.
            target_column: Target node column.
            graph_type: 'directed' or 'undirected'.
            weight_column: Optional weight column (ignored for path lengths).

        Returns:
            Dict with path statistics, clustering coefficient, and small-world σ.

        Raises:
            KeyError: If columns are not found.
            ValueError: If graph is too small or too large.
        """
        from lumen.statistics.graphs.network_density import AdjacencyMatrixBuilder, GraphType

        _VALID_TYPES: frozenset[str] = frozenset({"directed", "undirected"})

        for col in (source_column, target_column):
            if col not in edges.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if graph_type not in _VALID_TYPES:
            raise ValueError(
                f"graph_type must be one of {_VALID_TYPES}. Got '{graph_type}'."
            )

        gtype = GraphType(graph_type)
        adjacency, nodes = AdjacencyMatrixBuilder().build(
            edges, source_column, target_column, gtype, weight_column
        )

        if len(nodes) < self._MINIMUM_NODES:
            raise ValueError(
                f"At least {self._MINIMUM_NODES} nodes required. Got {len(nodes)}."
            )
        if len(nodes) > self._NODE_LIMIT:
            raise ValueError(
                f"Graph has {len(nodes)} nodes which exceeds the Floyd-Warshall "
                f"limit of {self._NODE_LIMIT}. Filter to a subgraph."
            )

        dist_matrix = self._shortest_paths.calculate(adjacency)
        path_stats = self._stats_extractor.extract(dist_matrix)
        clustering = self._clustering_calc.calculate(adjacency)

        sigma = None
        if path_stats["avg_shortest_path_length"] is not None:
            sigma = self._small_world_calc.calculate(
                clustering, path_stats["avg_shortest_path_length"], len(nodes)
            )

        return {
            "path_statistics": path_stats,
            "avg_clustering_coefficient": clustering,
            "small_world_coefficient": sigma,
            "small_world_interpretation": (
                "Small-world network detected." if sigma is not None and sigma > 1
                else "No significant small-world properties."
                if sigma is not None else None
            ),
            "n_nodes": len(nodes),
            "graph_type": graph_type,
        }