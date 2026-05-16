"""Network density, connectivity, and structural graph metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class GraphType(str, Enum):
    """Supported graph types."""

    DIRECTED = "directed"
    UNDIRECTED = "undirected"


@dataclass(frozen=True)
class GraphStructureResult:
    """Immutable structural metrics for a graph."""

    n_nodes: int
    n_edges: int
    density: float
    is_connected: bool
    n_connected_components: int
    largest_component_size: int
    largest_component_fraction: float
    avg_degree: float
    max_degree: int
    min_degree: int
    graph_type: str


class AdjacencyMatrixBuilder:
    """Builds an adjacency matrix from an edge list DataFrame.

    Supports directed and undirected graphs. Self-loops are excluded
    by zeroing the diagonal.
    """

    def build(
        self,
        edges: pd.DataFrame,
        source_column: str,
        target_column: str,
        graph_type: GraphType,
        weight_column: str | None,
    ) -> tuple[np.ndarray, list]:
        """Build adjacency matrix and ordered node list.

        Args:
            edges: DataFrame with source/target columns.
            source_column: Source node column name.
            target_column: Target node column name.
            graph_type: Directed or undirected.
            weight_column: Optional edge weight column.

        Returns:
            Tuple (adjacency_matrix, node_list).
        """
        all_nodes = sorted(
            set(edges[source_column].unique()) | set(edges[target_column].unique())
        )
        node_index = {node: i for i, node in enumerate(all_nodes)}
        n = len(all_nodes)
        matrix = np.zeros((n, n), dtype=float)

        for _, row in edges.iterrows():
            src = node_index[row[source_column]]
            tgt = node_index[row[target_column]]
            weight = float(row[weight_column]) if weight_column else 1.0
            matrix[src, tgt] += weight
            if graph_type == GraphType.UNDIRECTED:
                matrix[tgt, src] += weight

        np.fill_diagonal(matrix, 0)
        return matrix, all_nodes


class ConnectedComponentsFinder:
    """Finds connected components via BFS traversal on the adjacency matrix.

    For directed graphs, uses the symmetrized adjacency to find
    weakly connected components.
    """

    def find(self, adjacency: np.ndarray) -> tuple[int, list[set]]:
        """Find all connected components.

        Args:
            adjacency: Adjacency matrix (symmetrized for directed graphs).

        Returns:
            Tuple (n_components, list_of_component_node_sets).
        """
        n = adjacency.shape[0]
        visited = np.zeros(n, dtype=bool)
        components: list[set] = []

        # Symmetrize for weak connectivity
        sym = ((adjacency + adjacency.T) > 0).astype(int)

        for start in range(n):
            if visited[start]:
                continue
            component: set = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                visited[node] = True
                component.add(node)
                neighbors = np.where(sym[node] > 0)[0]
                queue.extend(int(nb) for nb in neighbors if not visited[nb])
            components.append(component)

        return len(components), components


class DegreeDistributionCalculator:
    """Computes degree statistics from adjacency matrix."""

    def calculate(
        self,
        adjacency: np.ndarray,
        graph_type: GraphType,
    ) -> dict:
        """Compute degree distribution statistics.

        Args:
            adjacency: Adjacency matrix.
            graph_type: Directed or undirected.

        Returns:
            Dict with degree stats and distribution array.
        """
        if graph_type == GraphType.DIRECTED:
            out_degree = (adjacency > 0).sum(axis=1).astype(float)
            in_degree = (adjacency > 0).sum(axis=0).astype(float)
            degrees = out_degree + in_degree
        else:
            degrees = (adjacency > 0).sum(axis=1).astype(float)

        return {
            "avg_degree": round(float(degrees.mean()), 4),
            "max_degree": int(degrees.max()),
            "min_degree": int(degrees.min()),
            "std_degree": round(float(degrees.std()), 4),
            "degree_values": degrees.tolist(),
        }


class NetworkDensityCalculator:
    """Structural graph metrics: density, connectivity, and degree distribution.

    Workflow:
        calculator = NetworkDensityCalculator()
        result = calculator.calculate(
            edges=df,
            source_column="from_node",
            target_column="to_node",
            graph_type="undirected",   # "directed" | "undirected"
            weight_column=None,        # optional
        )
    """

    _MINIMUM_EDGES: int = 1

    def __init__(self) -> None:
        self._matrix_builder = AdjacencyMatrixBuilder()
        self._component_finder = ConnectedComponentsFinder()
        self._degree_calculator = DegreeDistributionCalculator()

    def calculate(
        self,
        edges: pd.DataFrame,
        source_column: str,
        target_column: str,
        graph_type: str = "undirected",
        weight_column: str | None = None,
    ) -> dict:
        """Compute graph density and structural metrics.

        Args:
            edges: Edge list DataFrame.
            source_column: Source node column.
            target_column: Target node column.
            graph_type: 'directed' or 'undirected'.
            weight_column: Optional weight column.

        Returns:
            Dict with density, connectivity, degree stats, and component info.

        Raises:
            KeyError: If columns are not found.
            ValueError: If graph_type is invalid or edges are insufficient.
        """
        _VALID_TYPES: frozenset[str] = frozenset({"directed", "undirected"})

        for col in (source_column, target_column):
            if col not in edges.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if weight_column is not None and weight_column not in edges.columns:
            raise KeyError(f"Weight column '{weight_column}' not found.")
        if graph_type not in _VALID_TYPES:
            raise ValueError(
                f"graph_type must be one of {_VALID_TYPES}. Got '{graph_type}'."
            )
        if len(edges) < self._MINIMUM_EDGES:
            raise ValueError(
                f"At least {self._MINIMUM_EDGES} edge required. Got {len(edges)}."
            )

        gtype = GraphType(graph_type)
        adjacency, nodes = self._matrix_builder.build(
            edges, source_column, target_column, gtype, weight_column
        )
        n = len(nodes)
        n_edges = int((adjacency > 0).sum())
        if gtype == GraphType.UNDIRECTED:
            n_edges //= 2

        max_edges = n * (n - 1) if gtype == GraphType.DIRECTED else n * (n - 1) // 2
        density = n_edges / max_edges if max_edges > 0 else 0.0

        n_components, components = self._component_finder.find(adjacency)
        largest = max(components, key=len)
        is_connected = n_components == 1

        degree_stats = self._degree_calculator.calculate(adjacency, gtype)

        return {
            "n_nodes": n,
            "n_edges": n_edges,
            "density": round(density, 6),
            "density_interpretation": (
                "sparse" if density < 0.1
                else "moderate" if density < 0.5
                else "dense"
            ),
            "connectivity": {
                "is_connected": is_connected,
                "n_connected_components": n_components,
                "largest_component_size": len(largest),
                "largest_component_fraction": round(len(largest) / n, 4),
            },
            "degree_distribution": {
                "avg_degree": degree_stats["avg_degree"],
                "max_degree": degree_stats["max_degree"],
                "min_degree": degree_stats["min_degree"],
                "std_degree": degree_stats["std_degree"],
            },
            "graph_type": graph_type,
        }