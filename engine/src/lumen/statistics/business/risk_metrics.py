"""Financial risk metrics: VaR, CVaR, Sharpe, Sortino, and Calmar ratios."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `BusinessStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

@dataclass(frozen=True)
class RiskMetricResult:
    """Immutable result for a single risk metric."""

    metric_name: str
    value: float
    interpretation: str

class ReturnsComputer:
    """Computes simple or log returns from a price/value series.

    Simple return:  r(t) = (P(t) - P(t-1)) / P(t-1)
    Log return:     r(t) = ln(P(t) / P(t-1))

    Log returns are time-additive and preferred for multi-period analysis.
    """

    def compute(
        self, series: np.ndarray, method: str = "simple"
    ) -> np.ndarray:
        """Compute returns array.

        Args:
            series: Price or value time series (no NaN).
            method: 'simple' or 'log'.

        Returns:
            Returns array (length = len(series) - 1).

        Raises:
            ValueError: If method is invalid or series has non-positive values
                when using log returns.
        """
        if method not in ("simple", "log"):
            raise ValueError(
                f"method must be 'simple' or 'log'. Got '{method}'."
            )
        if method == "log":
            if np.any(series <= 0):
                raise ValueError(
                    "Log returns require strictly positive values. "
                    "Series contains non-positive values."
                )
            return np.diff(np.log(series))

        base = series[:-1]
        invalid = base == 0
        returns = np.where(
            invalid, np.nan, (series[1:] - base) / np.abs(base)
        )
        return returns

class VaRCalculator:
    """Value at Risk (VaR) via historical simulation.

    VaR(α) = -percentile(returns, α × 100)

    Represents the maximum loss not exceeded with probability (1-α).
    Example: VaR(0.05) = 3% means there is a 5% chance of losing more
    than 3% in a given period.
    """

    def calculate(
        self, returns: np.ndarray, confidence_level: float
    ) -> float:
        """Compute historical VaR.

        Args:
            returns: Array of period returns.
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR).

        Returns:
            VaR as a positive number representing the loss threshold.
        """
        alpha = 1.0 - confidence_level
        return float(-np.percentile(returns[~np.isnan(returns)], alpha * 100))

class CVaRCalculator:
    """Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR(α) = -mean(returns | returns < -VaR(α))

    Measures the expected loss given that we are in the worst (1-α)% of outcomes.
    Always >= VaR; more sensitive to tail risk.
    """

    def calculate(
        self, returns: np.ndarray, confidence_level: float
    ) -> float:
        """Compute historical CVaR.

        Args:
            returns: Array of period returns.
            confidence_level: Confidence level (same as VaR).

        Returns:
            CVaR as a positive number.
        """
        alpha = 1.0 - confidence_level
        clean = returns[~np.isnan(returns)]
        var_threshold = float(np.percentile(clean, alpha * 100))
        tail_returns = clean[clean <= var_threshold]
        return float(-tail_returns.mean()) if len(tail_returns) > 0 else 0.0

class SharpeRatioCalculator:
    """Sharpe ratio: risk-adjusted return above a risk-free rate.

    Sharpe = (μ_returns - r_f) / σ_returns × √(periods_per_year)

    Annualized by multiplying by √(periods_per_year).
    """

    def calculate(
        self,
        returns: np.ndarray,
        risk_free_rate: float,
        periods_per_year: int,
    ) -> float:
        """Compute annualized Sharpe ratio.

        Args:
            returns: Array of period returns.
            risk_free_rate: Annual risk-free rate (e.g., 0.02 = 2%).
            periods_per_year: Trading periods per year (252=daily, 12=monthly).

        Returns:
            Annualized Sharpe ratio.
        """
        clean = returns[~np.isnan(returns)]
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = clean - rf_per_period
        std = float(excess_returns.std(ddof=1))
        if std == 0.0:
            return 0.0
        return float(excess_returns.mean() / std * np.sqrt(periods_per_year))

class SortinoRatioCalculator:
    """Sortino ratio: like Sharpe but penalizes only downside volatility.

    Sortino = (μ_returns - r_f) / σ_downside × √(periods_per_year)

    σ_downside = std of returns below the target return (MAR).
    More appropriate when return distribution is asymmetric.
    """

    def calculate(
        self,
        returns: np.ndarray,
        risk_free_rate: float,
        periods_per_year: int,
        mar: float = 0.0,
    ) -> float:
        """Compute annualized Sortino ratio.

        Args:
            returns: Array of period returns.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Trading periods per year.
            mar: Minimum Acceptable Return per period (default 0).

        Returns:
            Annualized Sortino ratio.
        """
        clean = returns[~np.isnan(returns)]
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = clean - rf_per_period
        downside = clean[clean < mar] - mar
        downside_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0
        if downside_std == 0.0:
            return float("inf") if excess_returns.mean() > 0 else 0.0
        return float(excess_returns.mean() / downside_std * np.sqrt(periods_per_year))

class MaxDrawdownCalculator:
    """Maximum drawdown: largest peak-to-trough decline in portfolio value.

    MDD = (V_trough - V_peak) / V_peak

    Measures the worst historical loss from a peak, regardless of duration.
    """

    def calculate(self, series: np.ndarray) -> dict:
        """Compute maximum drawdown and its location.

        Args:
            series: Portfolio value or cumulative return series.

        Returns:
            Dict with mdd, peak_index, trough_index, and recovery_periods.
        """
        cummax = np.maximum.accumulate(series)
        drawdowns = (series - cummax) / cummax
        mdd_idx = int(np.argmin(drawdowns))
        peak_idx = int(np.argmax(series[:mdd_idx + 1])) if mdd_idx > 0 else 0

        # Find recovery: first period after trough where value >= peak
        peak_value = float(series[peak_idx])
        recovery_periods: int | None = None
        for j in range(mdd_idx + 1, len(series)):
            if series[j] >= peak_value:
                recovery_periods = j - mdd_idx
                break

        return {
            "max_drawdown": round(float(drawdowns[mdd_idx]), 6),
            "max_drawdown_pct": round(float(drawdowns[mdd_idx] * 100), 4),
            "peak_index": peak_idx,
            "trough_index": mdd_idx,
            "recovery_periods": recovery_periods,
        }

class CalmarRatioCalculator:
    """Calmar ratio: annualized return divided by maximum drawdown.

    Calmar = annualized_return / |MDD|

    Higher Calmar = better risk-adjusted performance relative to worst loss.
    """

    def calculate(
        self,
        returns: np.ndarray,
        max_drawdown: float,
        periods_per_year: int,
    ) -> float:
        """Compute Calmar ratio.

        Args:
            returns: Array of period returns.
            max_drawdown: Maximum drawdown (negative number or absolute).
            periods_per_year: Trading periods per year.

        Returns:
            Calmar ratio. Returns 0 if MDD is zero.
        """
        if max_drawdown == 0.0:
            return 0.0
        clean = returns[~np.isnan(returns)]
        annualized_return = float(clean.mean() * periods_per_year)
        return round(annualized_return / abs(max_drawdown), 4)

class RiskMetricsCalculator:
    """Full financial risk metrics suite.

    Workflow:
        calculator = RiskMetricsCalculator()
        result = calculator.calculate(
            data_frame=df,
            value_column="portfolio_value",   # price or cumulative value
            returns_column=None,              # optional, pre-computed returns
            returns_method="simple",          # "simple" | "log"
            confidence_level=0.95,
            risk_free_rate=0.02,
            periods_per_year=252,             # 252=daily, 12=monthly, 52=weekly
        )
    """

    _MINIMUM_PERIODS: int = 10

    def __init__(self) -> None:
        self._returns_computer = ReturnsComputer()
        self._var_calculator = VaRCalculator()
        self._cvar_calculator = CVaRCalculator()
        self._sharpe_calculator = SharpeRatioCalculator()
        self._sortino_calculator = SortinoRatioCalculator()
        self._mdd_calculator = MaxDrawdownCalculator()
        self._calmar_calculator = CalmarRatioCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        value_column: str,
        returns_column: str | None = None,
        returns_method: str = "simple",
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> dict:
        """Compute full risk metric suite.

        Args:
            data_frame: Time-ordered DataFrame.
            value_column: Portfolio value or price series column.
            returns_column: Pre-computed returns column (skips returns computation).
            returns_method: 'simple' or 'log' for return computation.
            confidence_level: VaR/CVaR confidence level (e.g., 0.95).
            risk_free_rate: Annual risk-free rate as decimal.
            periods_per_year: Periods per year for annualization.

        Returns:
            Dict with VaR, CVaR, Sharpe, Sortino, MDD, Calmar, and return stats.

        Raises:
            KeyError: If columns are not found.
            ValueError: If parameters are invalid or data is insufficient.
        """
        if value_column not in data_frame.columns:
            raise KeyError(f"Column '{value_column}' not found in DataFrame.")
        if returns_column is not None and returns_column not in data_frame.columns:
            raise KeyError(f"Column '{returns_column}' not found in DataFrame.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1). Got {confidence_level}."
            )
        if periods_per_year < 1:
            raise ValueError(
                f"periods_per_year must be >= 1. Got {periods_per_year}."
            )

        clean = data_frame[[value_column] + ([returns_column] if returns_column else [])].dropna()

        if len(clean) < self._MINIMUM_PERIODS:
            raise ValueError(
                f"At least {self._MINIMUM_PERIODS} observations required. "
                f"Got {len(clean)}."
            )

        values = clean[value_column].to_numpy(dtype=float)
        returns = (
            clean[returns_column].to_numpy(dtype=float)
            if returns_column is not None
            else self._returns_computer.compute(values, returns_method)
        )

        clean_returns = returns[~np.isnan(returns)]
        var = self._var_calculator.calculate(returns, confidence_level)
        cvar = self._cvar_calculator.calculate(returns, confidence_level)
        sharpe = self._sharpe_calculator.calculate(
            returns, risk_free_rate, periods_per_year
        )
        sortino = self._sortino_calculator.calculate(
            returns, risk_free_rate, periods_per_year
        )
        mdd_result = self._mdd_calculator.calculate(values)
        calmar = self._calmar_calculator.calculate(
            returns, mdd_result["max_drawdown"], periods_per_year
        )

        annualized_return = float(clean_returns.mean() * periods_per_year)
        annualized_vol = float(clean_returns.std(ddof=1) * np.sqrt(periods_per_year))

        return {
            "var": {
                "value": round(var, 6),
                "confidence_level": confidence_level,
                "interpretation": (
                    f"With {confidence_level*100:.0f}% confidence, the maximum "
                    f"single-period loss will not exceed {var*100:.2f}%."
                ),
            },
            "cvar": {
                "value": round(cvar, 6),
                "interpretation": (
                    f"When losses exceed VaR, the expected loss is {cvar*100:.2f}%."
                ),
            },
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": calmar,
            "max_drawdown": mdd_result,
            "return_statistics": {
                "annualized_return": round(annualized_return, 6),
                "annualized_volatility": round(annualized_vol, 6),
                "mean_period_return": round(float(clean_returns.mean()), 6),
                "std_period_return": round(float(clean_returns.std(ddof=1)), 6),
                "skewness": round(float(stats.skew(clean_returns)), 4),
                "excess_kurtosis": round(float(stats.kurtosis(clean_returns)), 4),
                "n_periods": len(clean_returns),
            },
            "parameters": {
                "risk_free_rate": risk_free_rate,
                "periods_per_year": periods_per_year,
                "returns_method": returns_method,
            },
        }
