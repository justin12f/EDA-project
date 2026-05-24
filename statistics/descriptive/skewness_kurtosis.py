"""Skewness and kurtosis analysis module."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `DescriptiveStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import numpy as np
from scipy import stats

class SkewnessInterpreter:
    """Classifies skewness severity and recommends transformations."""

    def interpret(self, skewness: float) -> dict[str, str]:
        """Interpret a skewness value.

        Args:
            skewness: Fisher-Pearson skewness coefficient.

        Returns:
            Dict with 'direction', 'severity', 'recommended_action'.
        """
        abs_skew = abs(skewness)

        if skewness > 0:
            direction = "right (positive)"
        elif skewness < 0:
            direction = "left (negative)"
        else:
            direction = "none"

        if abs_skew < 0.5:
            severity, action = "approximately symmetric", "no transformation needed"
        elif abs_skew < 1.0:
            severity = "moderately skewed"
            action = "consider sqrt or log transformation"
        else:
            severity = "highly skewed"
            action = "log1p or box_cox transformation recommended"

        return {
            "direction": direction,
            "severity": severity,
            "recommended_action": action,
        }

class KurtosisInterpreter:
    """Classifies excess kurtosis and advises on model selection."""

    def interpret(self, excess_kurtosis: float) -> dict[str, str]:
        """Interpret an excess kurtosis value.

        Args:
            excess_kurtosis: Fisher's excess kurtosis (normal = 0).

        Returns:
            Dict with 'distribution_type', 'recommended_action'.
        """
        if abs(excess_kurtosis) < 0.5:
            dist_type = "mesokurtic (normal-like tails)"
            action = "standard models safe"
        elif excess_kurtosis > 0.5:
            dist_type = "leptokurtic (heavy tails, outlier-prone)"
            action = "robust models or outlier handling recommended"
        else:
            dist_type = "platykurtic (light tails, fewer extremes)"
            action = "standard models generally safe"

        return {"distribution_type": dist_type, "recommended_action": action}

class SkewnessKurtosisCalculator:
    """Calculates skewness and kurtosis with actionable interpretations.

    Workflow:
        calculator = SkewnessKurtosisCalculator()
        result = calculator.calculate(data)

    Returns a dict with keys:
        - skewness, excess_kurtosis, pearson_kurtosis,
          skewness_interpretation, kurtosis_interpretation
    """

    _MINIMUM_SAMPLE_SIZE: int = 4

    def __init__(self) -> None:
        self._skewness_interpreter = SkewnessInterpreter()
        self._kurtosis_interpreter = KurtosisInterpreter()

    def calculate(self, data: np.ndarray, bias: bool = True) -> dict:
        """Calculate skewness and kurtosis.

        Args:
            data: 1D numerical array.
            bias: If True, uses biased estimator (scipy default).

        Returns:
            Dictionary with skewness, kurtosis, and interpretations.

        Raises:
            ValueError: If data has fewer than 4 elements.
        """
        if len(data) < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLE_SIZE} data points required. "
                f"Got {len(data)}."
            )

        skewness = float(stats.skew(data, bias=bias))
        excess_kurtosis = float(stats.kurtosis(data, bias=bias))  # Fisher (excess)
        pearson_kurtosis = excess_kurtosis + 3  # Pearson's definition

        return {
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "pearson_kurtosis": pearson_kurtosis,
            "skewness_interpretation": self._skewness_interpreter.interpret(skewness),
            "kurtosis_interpretation": self._kurtosis_interpreter.interpret(
                excess_kurtosis
            ),
        }
