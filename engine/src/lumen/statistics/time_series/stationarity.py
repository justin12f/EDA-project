"""Stationarity testing: ADF and KPSS with combined verdict."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `TimeSeriesStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class StationarityTestResult:
    """Immutable result for a single stationarity test."""

    test_name: str
    statistic: float
    p_value: float | None
    critical_values: dict[str, float]
    is_stationary: bool
    note: str

class AugmentedDickeyFullerTest:
    """ADF test for unit root — H0: series has a unit root (non-stationary)."""

    _CRITICAL_VALUES: dict[str, float] = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    _MINIMUM_OBSERVATIONS: int = 20
    _DEFAULT_LAGS: int = 1

    def test(self, series: np.ndarray, max_lags: int, significance_level: float) -> StationarityTestResult:
        n = len(series)
        lags = min(max_lags, n // 4)
        diff_series = np.diff(series)
        y = diff_series[lags:]
        x_cols = [np.ones(len(y)), series[lags: n - 1]]
        for lag in range(1, lags + 1):
            x_cols.append(diff_series[lags - lag: n - 1 - lag])
        x = np.column_stack(x_cols)
        coefficients, residuals_ss, _, _ = np.linalg.lstsq(x, y, rcond=None)
        df = len(y) - x.shape[1]
        if df <= 0 or len(residuals_ss) == 0:
            return StationarityTestResult(
                test_name="adf", statistic=0.0, p_value=None,
                critical_values=self._CRITICAL_VALUES, is_stationary=False,
                note="Insufficient degrees of freedom for ADF test.",
            )
        sigma_sq = float(residuals_ss[0]) / df if len(residuals_ss) > 0 else 1.0
        xtx_inv = np.linalg.pinv(x.T @ x)
        se_gamma = float(np.sqrt(sigma_sq * xtx_inv[1, 1]))
        t_stat = float(coefficients[1]) / se_gamma if se_gamma > 0 else 0.0
        threshold = self._CRITICAL_VALUES.get(f"{int(significance_level * 100)}%", self._CRITICAL_VALUES["5%"])
        is_stationary = t_stat < threshold
        return StationarityTestResult(
            test_name="adf", statistic=round(t_stat, 6), p_value=None,
            critical_values=self._CRITICAL_VALUES, is_stationary=is_stationary,
            note=f"H0: unit root. Stationary if t-stat < critical value. t={t_stat:.4f}, threshold={threshold}.",
        )

class KPSSTest:
    """KPSS test for stationarity — H0: series IS stationary."""

    _CRITICAL_VALUES: dict[str, float] = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}

    def test(self, series: np.ndarray, significance_level: float) -> StationarityTestResult:
        n = len(series)
        residuals = series - series.mean()
        partial_sums = np.cumsum(residuals)
        bandwidth = int(np.ceil(4 * (n / 100) ** 0.25))
        long_run_var = self._newey_west_variance(residuals, bandwidth)
        if long_run_var == 0.0:
            return StationarityTestResult(
                test_name="kpss", statistic=0.0, p_value=None,
                critical_values=self._CRITICAL_VALUES, is_stationary=True,
                note="Long-run variance is zero — series may be constant.",
            )
        kpss_stat = float(np.sum(partial_sums ** 2) / (n ** 2 * long_run_var))
        threshold = self._CRITICAL_VALUES.get(f"{int(significance_level * 100)}%", self._CRITICAL_VALUES["5%"])
        is_stationary = kpss_stat < threshold
        return StationarityTestResult(
            test_name="kpss", statistic=round(kpss_stat, 6), p_value=None,
            critical_values=self._CRITICAL_VALUES, is_stationary=is_stationary,
            note=f"H0: stationary. Reject stationarity if stat > critical value. stat={kpss_stat:.4f}, threshold={threshold}.",
        )

    def _newey_west_variance(self, residuals: np.ndarray, bandwidth: int) -> float:
        n = len(residuals)
        gamma_0 = float(np.sum(residuals ** 2) / n)
        weighted_sum = 0.0
        for lag in range(1, bandwidth + 1):
            gamma_j = float(np.sum(residuals[lag:] * residuals[:-lag]) / n)
            bartlett_weight = 1.0 - lag / (bandwidth + 1)
            weighted_sum += 2 * bartlett_weight * gamma_j
        return gamma_0 + weighted_sum

class StationarityVerdictInterpreter:
    """Combines ADF and KPSS results into a unified stationarity verdict."""

    def interpret(self, adf: StationarityTestResult, kpss: StationarityTestResult) -> str:
        if adf.is_stationary and kpss.is_stationary:
            return "stationary"
        if not adf.is_stationary and not kpss.is_stationary:
            return "non_stationary"
        return "inconclusive"

class StationarityCalculator:
    """Runs ADF + KPSS tests and returns a combined stationarity verdict."""

    _MINIMUM_OBSERVATIONS: int = 20

    def __init__(self) -> None:
        self._adf = AugmentedDickeyFullerTest()
        self._kpss = KPSSTest()
        self._interpreter = StationarityVerdictInterpreter()

    def calculate(self, series: np.ndarray, max_lags: int = 4, significance_level: float = 0.05) -> dict:
        if len(series) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(f"At least {self._MINIMUM_OBSERVATIONS} observations required. Got {len(series)}.")
        if max_lags < 1:
            raise ValueError(f"max_lags must be >= 1. Got {max_lags}.")

        adf_result = self._adf.test(series, max_lags, significance_level)
        kpss_result = self._kpss.test(series, significance_level)
        verdict = self._interpreter.interpret(adf_result, kpss_result)
        recommendation = {
            "stationary": "Series is ready for modelling. No differencing needed.",
            "non_stationary": "Apply first-order differencing (d=1) and re-test before modelling.",
            "inconclusive": "Results conflict. Inspect visually, consider seasonal adjustment or structural break tests.",
        }[verdict]

        return {
            "adf": {
                "statistic": adf_result.statistic,
                "critical_values": adf_result.critical_values,
                "is_stationary": adf_result.is_stationary,
                "note": adf_result.note,
            },
            "kpss": {
                "statistic": kpss_result.statistic,
                "critical_values": kpss_result.critical_values,
                "is_stationary": kpss_result.is_stationary,
                "note": kpss_result.note,
            },
            "combined_verdict": verdict,
            "recommendation": recommendation,
            "significance_level": significance_level,
            "n": len(series),
        }
