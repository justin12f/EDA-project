"""Frequency distribution builder module."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `DescriptiveStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import numpy as np

class BinCountSelector:
    """Selects an optimal bin count using classical rules.

    Supports:
        - 'sturges': log2(n)+1, best for normal-ish data.
        - 'scott': Based on std, good general purpose.
        - 'fd': Freedman-Diaconis, robust to outliers (uses IQR).
        - 'auto': Selects Sturges for n ≤ 200, FD otherwise.
    """

    def select(self, data: np.ndarray, method: str = "auto") -> int:
        """Select optimal bin count for the data.

        Args:
            data: 1D numerical array.
            method: One of 'sturges', 'scott', 'fd', 'auto'.

        Returns:
            Optimal integer bin count.
        """
        n = len(data)
        data_range = data.max() - data.min()

        if method == "sturges" or (method == "auto" and n <= 200):
            return max(1, int(np.ceil(np.log2(n))) + 1)

        if method == "scott":
            h = 3.5 * float(np.std(data)) / (n ** (1 / 3))
            return max(1, int(np.ceil(data_range / h))) if h > 0 else 10

        # Freedman-Diaconis — default for large n
        iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
        h = 2 * iqr / (n ** (1 / 3))
        return max(1, int(np.ceil(data_range / h))) if h > 0 else 10

class FrequencyTableBuilder:
    """Builds a frequency distribution table from histogram data."""

    def build(self, data: np.ndarray, n_bins: int) -> list[dict]:
        """Build frequency table rows.

        Args:
            data: 1D numerical array.
            n_bins: Number of bins.

        Returns:
            List of dicts: bin_start, bin_end, bin_label, frequency,
            relative_frequency, cumulative_frequency.
        """
        counts, bin_edges = np.histogram(data, bins=n_bins)
        n_total = len(data)
        relative_frequencies = counts / n_total
        cumulative_frequencies = np.cumsum(relative_frequencies)

        return [
            {
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "bin_label": f"[{bin_edges[i]:.4g}, {bin_edges[i + 1]:.4g})",
                "frequency": int(counts[i]),
                "relative_frequency": float(relative_frequencies[i]),
                "cumulative_frequency": float(cumulative_frequencies[i]),
            }
            for i in range(len(counts))
        ]

class FrequencyDistributionBuilder:
    """Orchestrates full frequency distribution construction.

    Workflow:
        builder = FrequencyDistributionBuilder()
        result = builder.build(data, n_bins=None, bin_method="auto")

    Returns a dict with keys:
        - table, n_bins, total_count, bin_method
    """

    def __init__(self) -> None:
        self._bin_selector = BinCountSelector()
        self._table_builder = FrequencyTableBuilder()

    def build(
        self,
        data: np.ndarray,
        n_bins: int | None = None,
        bin_method: str = "auto",
    ) -> dict:
        """Build the complete frequency distribution.

        Args:
            data: 1D numerical array.
            n_bins: Number of bins. Auto-selected if None.
            bin_method: Method for auto selection: 'sturges', 'scott', 'fd', 'auto'.

        Returns:
            Dictionary with frequency table and metadata.

        Raises:
            ValueError: If data is empty.
        """
        if len(data) == 0:
            raise ValueError("Data array cannot be empty.")

        actual_bins = (
            n_bins
            if n_bins is not None
            else self._bin_selector.select(data, bin_method)
        )

        return {
            "table": self._table_builder.build(data, actual_bins),
            "n_bins": actual_bins,
            "total_count": len(data),
            "bin_method": bin_method if n_bins is None else "manual",
        }
