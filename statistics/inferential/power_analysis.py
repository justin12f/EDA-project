"""Statistical power analysis module."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `InferentialStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import numpy as np
from scipy import stats, optimize

class MinimumSampleSizeCalculator:
    """Calculates minimum n per group to achieve target power.

    Uses normal approximation for the two-sample T-test.
    Solves numerically for n given alpha, power, and Cohen's d.
    """

    def calculate(
        self,
        effect_size: float,
        alpha: float,
        target_power: float,
    ) -> int:
        """Compute minimum sample size per group.

        Args:
            effect_size: Expected Cohen's d.
            alpha: Type I error rate.
            target_power: Desired statistical power (1 - beta).

        Returns:
            Minimum n per group (always rounded up).

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if effect_size <= 0:
            raise ValueError(f"effect_size must be > 0. Got {effect_size}.")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1). Got {alpha}.")
        if not 0 < target_power < 1:
            raise ValueError(
                f"target_power must be in (0, 1). Got {target_power}."
            )

        z_alpha = float(stats.norm.ppf(1 - alpha / 2))
        z_beta = float(stats.norm.ppf(target_power))
        n = ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

class ObservedPowerCalculator:
    """Calculates observed power for a completed two-sample T-test.

    Given the actual sample sizes and effect size, estimates the
    probability of detecting the effect if it truly exists.
    """

    def calculate(
        self,
        effect_size: float,
        n_per_group: int,
        alpha: float,
    ) -> float:
        """Compute observed statistical power.

        Args:
            effect_size: Observed Cohen's d.
            n_per_group: Observations per group.
            alpha: Type I error rate used in the test.

        Returns:
            Power value in [0, 1].

        Raises:
            ValueError: If n_per_group < 2 or alpha out of range.
        """
        if n_per_group < 2:
            raise ValueError(
                f"n_per_group must be ≥ 2. Got {n_per_group}."
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1). Got {alpha}.")

        z_alpha = float(stats.norm.ppf(1 - alpha / 2))
        ncp = abs(effect_size) * float(np.sqrt(n_per_group / 2))
        power = float(1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp))
        return min(max(power, 0.0), 1.0)

class PowerAnalysisCalculator:
    """Unified power analysis dispatcher.

    Workflow:
        calculator = PowerAnalysisCalculator()

        # Minimum sample size
        result = calculator.calculate(
            "minimum_n",
            effect_size=0.5,
            alpha=0.05,
            target_power=0.80,
        )

        # Observed power
        result = calculator.calculate(
            "observed_power",
            effect_size=0.5,
            n_per_group=64,
            alpha=0.05,
        )
    """

    def __init__(self) -> None:
        self._minimum_n = MinimumSampleSizeCalculator()
        self._observed_power = ObservedPowerCalculator()

    def calculate(self, analysis_type: str, **kwargs) -> dict:
        """Dispatch power analysis calculation.

        Args:
            analysis_type: One of 'minimum_n', 'observed_power'.
            **kwargs: Arguments forwarded to the specific calculator.

        Returns:
            Dictionary with result and all input parameters.

        Raises:
            KeyError: If analysis_type is not recognized.
        """
        if analysis_type == "minimum_n":
            n = self._minimum_n.calculate(
                effect_size=kwargs["effect_size"],
                alpha=kwargs["alpha"],
                target_power=kwargs["target_power"],
            )
            return {
                "analysis_type": "minimum_n",
                "minimum_n_per_group": n,
                "total_n": n * 2,
                "effect_size": kwargs["effect_size"],
                "alpha": kwargs["alpha"],
                "target_power": kwargs["target_power"],
            }

        if analysis_type == "observed_power":
            power = self._observed_power.calculate(
                effect_size=kwargs["effect_size"],
                n_per_group=kwargs["n_per_group"],
                alpha=kwargs["alpha"],
            )
            return {
                "analysis_type": "observed_power",
                "observed_power": power,
                "effect_size": kwargs["effect_size"],
                "n_per_group": kwargs["n_per_group"],
                "alpha": kwargs["alpha"],
                "is_adequately_powered": power >= 0.80,
            }

        raise KeyError(
            f"analysis_type '{analysis_type}' not recognized. "
            f"Available: 'minimum_n', 'observed_power'."
        )
