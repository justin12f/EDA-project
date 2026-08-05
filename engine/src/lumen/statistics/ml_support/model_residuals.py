"""Regression residual diagnostics: normality, homoscedasticity, autocorrelation."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `MlSupportStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

@dataclass(frozen=True)
class ResidualDiagnosticResult:
    """Immutable result for a single residual diagnostic check."""

    check_name: str
    passed: bool
    statistic: float
    p_value: float | None
    interpretation: str

class ResidualNormalityChecker:
    """Tests whether residuals follow a normal distribution via Shapiro-Wilk.

    Normality of residuals is an assumption of OLS inference.
    Violations suggest the model may be misspecified or that
    transformation of the target variable is needed.
    """

    _MAX_SAMPLE: int = 5_000

    def check(
        self, residuals: np.ndarray, significance_level: float
    ) -> ResidualDiagnosticResult:
        """Run Shapiro-Wilk normality test on residuals.

        Args:
            residuals: OLS residual array.
            significance_level: Alpha threshold.

        Returns:
            ResidualDiagnosticResult for normality check.
        """
        sample = residuals[:self._MAX_SAMPLE]
        statistic, p_value = stats.shapiro(sample)
        passed = float(p_value) > significance_level

        return ResidualDiagnosticResult(
            check_name="normality_shapiro_wilk",
            passed=passed,
            statistic=round(float(statistic), 6),
            p_value=round(float(p_value), 6),
            interpretation=(
                "Residuals are approximately normal — OLS inference is valid."
                if passed else
                "Residuals deviate from normality. Consider target transformation "
                "or robust standard errors."
            ),
        )

class HomoscedasticityChecker:
    """Breusch-Pagan test for heteroscedasticity.

    Regresses squared residuals on fitted values and tests whether
    the relationship is significant. A significant result means the
    error variance is not constant across the range of fitted values.

    H0: homoscedastic (constant variance).
    H1: heteroscedastic (variance depends on fitted values).
    """

    def check(
        self,
        residuals: np.ndarray,
        y_pred: np.ndarray,
        significance_level: float,
    ) -> ResidualDiagnosticResult:
        """Run Breusch-Pagan test via OLS regression of squared residuals.

        Args:
            residuals: OLS residual array.
            y_pred: Fitted (predicted) values.
            significance_level: Alpha threshold.

        Returns:
            ResidualDiagnosticResult for homoscedasticity check.
        """
        sq_residuals = residuals ** 2
        x = np.column_stack([np.ones(len(y_pred)), y_pred])

        coefficients, *_ = np.linalg.lstsq(x, sq_residuals, rcond=None)
        fitted_sq = x @ coefficients
        ss_reg = float(np.sum((fitted_sq - sq_residuals.mean()) ** 2))
        ss_tot = float(np.sum((sq_residuals - sq_residuals.mean()) ** 2))

        r_squared = ss_reg / ss_tot if ss_tot > 0 else 0.0
        n = len(residuals)
        bp_statistic = n * r_squared
        p_value = float(1.0 - stats.chi2.cdf(bp_statistic, df=1))
        passed = p_value > significance_level

        return ResidualDiagnosticResult(
            check_name="homoscedasticity_breusch_pagan",
            passed=passed,
            statistic=round(bp_statistic, 6),
            p_value=round(p_value, 6),
            interpretation=(
                "No significant heteroscedasticity detected — variance appears constant."
                if passed else
                "Heteroscedasticity detected. Consider WLS, log-transform of target, "
                "or robust standard errors."
            ),
        )

class AutocorrelationChecker:
    """Durbin-Watson test for first-order autocorrelation in residuals.

    DW ≈ 2: no autocorrelation.
    DW < 1.5: positive autocorrelation (residuals are trending).
    DW > 2.5: negative autocorrelation (residuals oscillate).

    Critical values are approximated from the standard rule of thumb.
    """

    _LOWER_BOUND: float = 1.5
    _UPPER_BOUND: float = 2.5

    def check(self, residuals: np.ndarray) -> ResidualDiagnosticResult:
        """Compute Durbin-Watson statistic.

        Args:
            residuals: OLS residual array (time-ordered).

        Returns:
            ResidualDiagnosticResult for autocorrelation check.
        """
        diffs = np.diff(residuals)
        dw = float(np.sum(diffs ** 2) / np.sum(residuals ** 2))
        passed = self._LOWER_BOUND <= dw <= self._UPPER_BOUND

        if dw < self._LOWER_BOUND:
            interpretation = (
                f"DW={dw:.4f}: positive autocorrelation detected. "
                "Consider adding lag features or using GLS/HAC standard errors."
            )
        elif dw > self._UPPER_BOUND:
            interpretation = (
                f"DW={dw:.4f}: negative autocorrelation detected. "
                "Model may be over-differenced or have seasonal misspecification."
            )
        else:
            interpretation = f"DW={dw:.4f}: no significant autocorrelation detected."

        return ResidualDiagnosticResult(
            check_name="autocorrelation_durbin_watson",
            passed=passed,
            statistic=round(dw, 6),
            p_value=None,
            interpretation=interpretation,
        )

class ModelResidualsCalculator:
    """Full residual diagnostic suite: normality, homoscedasticity, autocorrelation.

    Workflow:
        calculator = ModelResidualsCalculator()
        result = calculator.calculate(
            y_true=df["actual"].to_numpy(),
            y_pred=df["predicted"].to_numpy(),
            significance_level=0.05,
        )
    """

    _MINIMUM_OBSERVATIONS: int = 8

    def __init__(self) -> None:
        self._normality = ResidualNormalityChecker()
        self._homoscedasticity = HomoscedasticityChecker()
        self._autocorrelation = AutocorrelationChecker()

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        significance_level: float = 0.05,
    ) -> dict:
        """Run all residual diagnostics.

        Args:
            y_true: Actual target values.
            y_pred: Model predicted values.
            significance_level: Alpha threshold for significance tests.

        Returns:
            Dict with per-check results, overall pass/fail, and residual stats.

        Raises:
            ValueError: If arrays have different lengths or are too short.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have equal length. "
                f"Got {len(y_true)} and {len(y_pred)}."
            )
        if len(y_true) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {len(y_true)}."
            )
        if not 0.0 < significance_level < 1.0:
            raise ValueError(
                f"significance_level must be in (0, 1). Got {significance_level}."
            )

        residuals = y_true - y_pred

        normality = self._normality.check(residuals, significance_level)
        homoscedasticity = self._homoscedasticity.check(
            residuals, y_pred, significance_level
        )
        autocorrelation = self._autocorrelation.check(residuals)

        checks = [normality, homoscedasticity, autocorrelation]
        all_passed = all(c.passed for c in checks)

        return {
            "checks": {
                c.check_name: {
                    "passed": c.passed,
                    "statistic": c.statistic,
                    "p_value": c.p_value,
                    "interpretation": c.interpretation,
                }
                for c in checks
            },
            "all_checks_passed": all_passed,
            "residual_summary": {
                "mean": round(float(residuals.mean()), 6),
                "std": round(float(residuals.std(ddof=1)), 6),
                "min": round(float(residuals.min()), 6),
                "max": round(float(residuals.max()), 6),
                "p5": round(float(np.percentile(residuals, 5)), 6),
                "p95": round(float(np.percentile(residuals, 95)), 6),
            },
            "n_observations": len(y_true),
            "significance_level": significance_level,
        }
