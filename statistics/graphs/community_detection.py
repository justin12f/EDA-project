"""Community detection via Louvain-style modularity optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Community:
    """Immutable community record."""

    community_id: int
    members: list[str]
    n_members: int
    proportion: float
    internal_edge_density: float


class ModularityCalculator:
    """Computes Newman-Girvan modularity Q for a given partition.

    Q = (1/2m) × Σ_{ij} [A_{ij} - k_i×k_j/(2m)] × δ(c_i, c_j)

    where m = total edges, k_i = degree of i, δ = 1 if same community.
    Q ∈ [-0.5, 1.0]; Q > 0.3 typically indicates meaningful structure.
    """

    def calculate(
        self,
        adjacency: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute modularity Q.

        Args:
            adjacency: Symmetric adjacency matrix.
            labels: Community label array (one per node).

        Returns:
            Modularity Q value.
        """
        m = float(adjacency.sum()) / 2
        if m == 0:
            return 0.0

        degrees = adjacency.sum(axis=1)
        n = adjacency.shape[0]
        q = 0.0

        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    expected = degrees[i] * degrees[j] / (2 * m)
                    q += adjacency[i, j] - expected

        return float(q / (2 * m))


class GreedyModularityOptimizer:
    """Greedy modularity maximization for community detection.

    Approximates Louvain algorithm:
    1. Start with each node in its own community.
    2. For each node, try merging with each neighbor's community.
    3. Keep the merge that maximizes modularity gain.
    4. Repeat until no improvement.

    This is a simplified O(n²) version. For production with large graphs
    (n > 1000), use the full Louvain with community library.
    """

    def __init__(self, modularity_calc: ModularityCalculator) -> None:
        self._modularity_calc = modularity_calc

    def optimize(
        self,
        adjacency: np.ndarray,
        random_seed: int,
    ) -> np.ndarray:
        """Run greedy modularity optimization.

        Args:
            adjacency: Symmetric adjacency matrix.
            random_seed: Seed for tie-breaking.

        Returns:
            Community label array (integers starting at 0).
        """
        rng = np.random.default_rng(random_seed)
        n = adjacency.shape[0]
        labels = np.arange(n)
        best_q = self._modularity_calc.calculate(adjacency, labels)

        improved = True
        max_passes = 10  # safety cap

        while improved and max_passes > 0:
            improved = False
            max_passes -= 1
            node_order = rng.permutation(n)

            for node in node_order:
                current_label = labels[node]
                neighbor_labels = set(
                    labels[j] for j in range(n)
                    if adjacency[node, j] > 0 and j != node
                )

                best_label = current_label
                best_q_local = best_q

                for candidate_label in neighbor_labels:
                    if candidate_label == current_label:
                        continue
                    trial_labels = labels.copy()
                    trial_labels[node] = candidate_label
                    q_trial = self._modularity_calc.calculate(adjacency, trial_labels)

                    if q_trial > best_q_local:
                        best_q_local = q_trial
                        best_label = candidate_label

                if best_label != current_label:
                    labels[node] = best_label
                    best_q = best_q_local
                    improved = True

        # Relabel communities to consecutive integers starting at 0
        unique_labels = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        return np.array([unique_labels[l] for l in labels])


class CommunityProfileBuilder:
    """Builds statistical profiles for each detected community."""

    def build(
        self,
        adjacency: np.ndarray,
        labels: np.ndarray,
        nodes: list,
    ) -> list[Community]:
        """Build community profiles with internal edge density.

        Args:
            adjacency: Adjacency matrix.
            labels: Community label array.
            nodes: Ordered node label list.

        Returns:
            List of Community objects sorted by size descending.
        """
        n_total = len(nodes)
        unique_labels = sorted(set(labels))
        communities: list[Community] = []

        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]
            members = [str(nodes[i]) for i in indices]
            n_members = len(members)

            # Internal edge density
            submatrix = adjacency[np.ix_(indices, indices)]
            n_possible = n_members * (n_members - 1)
            internal_edges = float((submatrix > 0).sum())
            if n_members > 1:
                internal_edges /= 2  # undirected
            internal_density = internal_edges / (n_possible / 2) if n_possible > 0 else 0.0

            communities.append(
                Community(
                    community_id=int(label),
                    members=members,
                    n_members=n_members,
                    proportion=round(n_members / n_total, 4),
                    internal_edge_density=round(internal_density, 4),
                )
            )

        return sorted(communities, key=lambda c: c.n_members, reverse=True)


class CommunityDetectionCalculator:
    """Louvain-style greedy community detection with modularity scoring.

    Workflow:
        calculator = CommunityDetectionCalculator()
        result = calculator.calculate(
            edges=df,
            source_column="from_node",
            target_column="to_node",
            random_seed=42,
        )
    """

    _MINIMUM_NODES: int = 4
    _NODE_LIMIT: int = 500

    def __init__(self) -> None:
        self._modularity_calc = ModularityCalculator()
        self._optimizer = GreedyModularityOptimizer(self._modularity_calc)
        self._profile_builder = CommunityProfileBuilder()

    def calculate(
        self,
        edges: pd.DataFrame,
        source_column: str,
        target_column: str,
        random_seed: int = 42,
        weight_column: str | None = None,
    ) -> dict:
        """Detect communities via greedy modularity optimization.

        Args:
            edges: Edge list DataFrame.
            source_column: Source node column.
            target_column: Target node column.
            random_seed: Seed for optimization reproducibility.
            weight_column: Optional weight column.

        Returns:
            Dict with community profiles, modularity Q, and node assignments.

        Raises:
            KeyError: If columns are not found.
            ValueError: If graph is too small or too large for detection.
        """
        from statistics.graphs.network_density import AdjacencyMatrixBuilder, GraphType

        for col in (source_column, target_column):
            if col not in edges.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        adjacency, nodes = AdjacencyMatrixBuilder().build(
            edges, source_column, target_column, GraphType.UNDIRECTED, weight_column
        )

        if len(nodes) < self._MINIMUM_NODES:
            raise ValueError(
                f"At least {self._MINIMUM_NODES} nodes required. Got {len(nodes)}."
            )
        if len(nodes) > self._NODE_LIMIT:
            raise ValueError(
                f"Graph has {len(nodes)} nodes which exceeds the limit of "
                f"{self._NODE_LIMIT} for greedy community detection. "
                "Consider filtering to a subgraph or using networkx for larger graphs."
            )

        labels = self._optimizer.optimize(adjacency, random_seed)
        modularity = self._modularity_calc.calculate(adjacency, labels)
        communities = self._profile_builder.build(adjacency, labels, nodes)

        node_assignments = {
            str(nodes[i]): int(labels[i]) for i in range(len(nodes))
        }

        return {
            "communities": [
                {
                    "community_id": c.community_id,
                    "n_members": c.n_members,
                    "proportion": c.proportion,
                    "members": c.members,
                    "internal_edge_density": c.internal_edge_density,
                }
                for c in communities
            ],
            "node_community_assignments": node_assignments,
            "modularity_q": round(modularity, 6),
            "modularity_interpretation": (
                "strong community structure" if modularity > 0.3
                else "moderate community structure" if modularity > 0.1
                else "weak or no community structure"
            ),
            "n_communities": len(communities),
            "n_nodes": len(nodes),
            "random_seed": random_seed,
        }