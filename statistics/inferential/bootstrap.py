"""Non-parametric bootstrap estimation module."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `InferentialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from typing import Callable

import numpy as np

class BootstrapSampler:
    """Generates bootstrap samples with a fixed random seed for reproducibility."""

    def generate(
        self,
        data: np.ndarray,
        n_iterations: int,
        random_seed: int | None,
    ) -> np.ndarray:
        """Generate a 2D array of bootstrap samples.

        Args:
            data: 1D source array.
            n_iterations: Number of bootstrap resamples.
            random_seed: Seed for reproducibility. None = non-deterministic.

        Returns:
            2D array of shape (n_iterations, len(data)).
        """
        rng = np.random.default_rng(random_seed)
        return rng.choice(data, size=(n_iterations, len(data)), replace=True)

class BootstrapStatisticEstimator:
    """Applies a statistic function to each bootstrap sample row."""

    def estimate(
        self,
        bootstrap_samples: np.ndarray,
        statistic: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        """Apply statistic to each bootstrap sample.

        Args:
            bootstrap_samples: 2D array (n_iterations × n).
            statistic: Callable that takes a 1D array and returns a scalar.

        Returns:
            1D array of bootstrap statistic values.

        Raises:
            TypeError: If statistic is not callable.
        """
        if not callable(statistic):
            raise TypeError(
                f"statistic must be callable. Got {type(statistic).__name__}."
            )

        return np.array([statistic(row) for row in bootstrap_samples])

class PercentilesBootstrapCI:
    """Calculates the percentile bootstrap confidence interval."""

    def calculate(
        self,
        bootstrap_statistics: np.ndarray,
        confidence_level: float,
    ) -> tuple[float, float]:
        """Calculate percentile CI from bootstrap distribution.

        Args:
            bootstrap_statistics: 1D array of bootstrap statistic values.
            confidence_level: Desired confidence level (e.g., 0.95).

        Returns:
            Tuple (lower_bound, upper_bound).
        """
        alpha = 1.0 - confidence_level
        lower = float(np.percentile(bootstrap_statistics, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2)))
        return lower, upper

class BootstrapEstimator:
    """Non-parametric bootstrap CI for any scalar statistic.

    Workflow:
        estimator = BootstrapEstimator()
        result = estimator.estimate(
            data=arr,
            statistic=np.median,
            n_iterations=5000,
            confidence_level=0.95,
            random_seed=42,
        )
    """

    _MINIMUM_SAMPLE_SIZE: int = 10
    _MINIMUM_ITERATIONS: int = 100

    def __init__(self) -> None:
        self._sampler = BootstrapSampler()
        self._estimator = BootstrapStatisticEstimator()
        self._ci_calculator = PercentilesBootstrapCI()

    def estimate(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        n_iterations: int = 5_000,
        confidence_level: float = 0.95,
        random_seed: int | None = 42,
    ) -> dict:
        """Run bootstrap estimation for a given statistic.

        Args:
            data: 1D numerical array.
            statistic: Any callable returning a scalar (e.g., np.mean, np.median).
            n_iterations: Number of bootstrap resamples.
            confidence_level: Desired CI confidence level.
            random_seed: Seed for reproducibility.

        Returns:
            Dictionary with observed statistic, CI, bootstrap distribution stats.

        Raises:
            ValueError: If data or iteration count is too small.
            TypeError: If statistic is not callable.
        """
        if len(data) < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"Bootstrap requires at least {self._MINIMUM_SAMPLE_SIZE} "
                f"observations. Got {len(data)}."
            )
        if n_iterations < self._MINIMUM_ITERATIONS:
            raise ValueError(
                f"n_iterations must be ≥ {self._MINIMUM_ITERATIONS}. "
                f"Got {n_iterations}."
            )
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1). Got {confidence_level}."
            )

        observed = float(statistic(data))
        samples = self._sampler.generate(data, n_iterations, random_seed)
        bootstrap_dist = self._estimator.estimate(samples, statistic)
        lower, upper = self._ci_calculator.calculate(bootstrap_dist, confidence_level)

        return {
            "observed_statistic": observed,
            "confidence_interval": {
                "lower": lower,
                "upper": upper,
                "confidence_level": confidence_level,
            },
            "bootstrap_distribution": {
                "mean": float(bootstrap_dist.mean()),
                "std": float(bootstrap_dist.std()),
                "bias": float(bootstrap_dist.mean() - observed),
            },
            "n_iterations": n_iterations,
            "n": len(data),
            "random_seed": random_seed,
        }
