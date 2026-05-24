"""Central tendency measures module."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `DescriptiveStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import numpy as np
from scipy import stats

class MeanCalculator:
    """Arithmetic mean."""

    def calculate(self, data: np.ndarray) -> float:
        return float(np.mean(data))

class MedianCalculator:
    """Median — 50th percentile, robust to outliers."""

    def calculate(self, data: np.ndarray) -> float:
        return float(np.median(data))

class ModeCalculator:
    """Mode — most frequent value. Returns value and its count."""

    def calculate(self, data: np.ndarray) -> dict[str, float]:
        mode_result = stats.mode(data, keepdims=True)
        return {
            "value": float(mode_result.mode[0]),
            "count": int(mode_result.count[0]),
        }

class TrimmedMeanCalculator:
    """Trimmed mean — mean after removing extreme proportions on both ends."""

    def calculate(self, data: np.ndarray, trim_proportion: float) -> float:
        return float(stats.trim_mean(data, trim_proportion))

class CentralTendencyInterpreter:
    """Interprets the relationship between mean and median to hint at skewness."""

    _SYMMETRY_THRESHOLD: float = 0.05

    def interpret(self, mean: float, median: float) -> str:
        """Classify the distribution shape based on mean-median relationship.

        Args:
            mean: Arithmetic mean.
            median: Median.

        Returns:
            One of: 'symmetric', 'right_skewed', 'left_skewed', 'mean_is_zero'.
        """
        if mean == 0:
            return "mean_is_zero"

        relative_difference = abs(mean - median) / abs(mean)

        if relative_difference < self._SYMMETRY_THRESHOLD:
            return "symmetric"
        return "right_skewed" if mean > median else "left_skewed"

class CentralTendencyCalculator:
    """Calculates all central tendency measures with interpretation.

    Workflow:
        calculator = CentralTendencyCalculator()
        result = calculator.calculate(data, trim_proportion=0.1)

    Returns a dict with keys:
        - mean, median, mode, trimmed_mean, trim_proportion,
          distribution_shape_hint
    """

    def __init__(self) -> None:
        self._mean_calc = MeanCalculator()
        self._median_calc = MedianCalculator()
        self._mode_calc = ModeCalculator()
        self._trimmed_mean_calc = TrimmedMeanCalculator()
        self._interpreter = CentralTendencyInterpreter()

    def calculate(self, data: np.ndarray, trim_proportion: float = 0.1) -> dict:
        """Calculate all central tendency measures.

        Args:
            data: 1D numerical array.
            trim_proportion: Proportion trimmed from each end for trimmed mean.
                Must be in [0.0, 0.5).

        Returns:
            Dictionary with all central tendency measures.

        Raises:
            ValueError: If data is empty or trim_proportion is out of range.
        """
        if len(data) == 0:
            raise ValueError("Data array cannot be empty.")

        if not 0.0 <= trim_proportion < 0.5:
            raise ValueError("trim_proportion must be in the range [0.0, 0.5).")

        mean = self._mean_calc.calculate(data)
        median = self._median_calc.calculate(data)
        mode = self._mode_calc.calculate(data)
        trimmed_mean = self._trimmed_mean_calc.calculate(data, trim_proportion)
        shape_hint = self._interpreter.interpret(mean, median)

        return {
            "mean": mean,
            "median": median,
            "mode": mode,
            "trimmed_mean": trimmed_mean,
            "trim_proportion": trim_proportion,
            "distribution_shape_hint": shape_hint,
        }
