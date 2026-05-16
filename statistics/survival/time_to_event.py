"""Time-to-event descriptive statistics and threshold analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class TimeToEventSummary:
    """Immutable time-to-event summary statistics."""

    n_total: int
    n_events: int
    n_censored: int
    event_rate: float
    mean_time: float
    median_time: float
    std_time: float
    p25_time: float
    p75_time: float
    min_time: float
    max_time: float
    iqr_time: float


class TimeToEventSummaryCalculator:
    """Computes descriptive statistics for time-to-event data."""

    def calculate(
        self,
        times: np.ndarray,
        events: np.ndarray,
    ) -> TimeToEventSummary:
        """Compute summary statistics for the full cohort.

        Args:
            times: Time array (all observations including censored).
            events: Event indicator (1=event, 0=censored).

        Returns:
            TimeToEventSummary dataclass.
        """
        n_events = int(events.sum())
        n_total = len(times)
        event_times = times[events == 1]

        return TimeToEventSummary(
            n_total=n_total,
            n_events=n_events,
            n_censored=n_total - n_events,
            event_rate=round(n_events / n_total, 4),
            mean_time=round(float(times.mean()), 4),
            median_time=round(float(np.median(times)), 4),
            std_time=round(float(times.std(ddof=1)), 4),
            p25_time=round(float(np.percentile(times, 25)), 4),
            p75_time=round(float(np.percentile(times, 75)), 4),
            min_time=round(float(times.min()), 4),
            max_time=round(float(times.max()), 4),
            iqr_time=round(float(np.percentile(times, 75) - np.percentile(times, 25)), 4),
        )


class ThresholdSurvivalAnalyser:
    """Estimates fraction of subjects surviving past specific time thresholds.

    Uses the empirical survival function — counts subjects with
    time >= threshold regardless of censoring status.
    """

    def analyse(
        self,
        times: np.ndarray,
        thresholds: list[float],
    ) -> list[dict]:
        """Compute threshold-based survival fractions.

        Args:
            times: Time array.
            thresholds: List of time threshold values.

        Returns:
            List of dicts with threshold, n_survived, and fraction.
        """
        n_total = len(times)
        return [
            {
                "threshold": threshold,
                "n_survived": int(np.sum(times >= threshold)),
                "fraction_survived": round(float(np.sum(times >= threshold)) / n_total, 4),
                "fraction_not_survived": round(float(np.sum(times < threshold)) / n_total, 4),
            }
            for threshold in sorted(thresholds)
        ]


class ExponentialFitter:
    """Fits an exponential survival model: S(t) = exp(-λt).

    MLE estimate: λ̂ = d / Σt  where d = events, Σt = total time.
    Valid when hazard rate is approximately constant (memoryless property).
    """

    def fit(
        self,
        times: np.ndarray,
        events: np.ndarray,
    ) -> dict:
        """Fit exponential model via MLE.

        Args:
            times: Time array.
            events: Event indicator array.

        Returns:
            Dict with rate lambda, mean survival time, and goodness-of-fit.
        """
        n_events = float(events.sum())
        total_time = float(times.sum())

        if n_events == 0 or total_time == 0:
            return {
                "lambda_rate": None,
                "mean_survival_time": None,
                "note": "Cannot fit: no events or zero total time.",
            }

        lambda_mle = n_events / total_time

        # Test exponential goodness-of-fit via KS test on event times
        event_times = times[events == 1]
        if len(event_times) > 3:
            ks_stat, ks_p = stats.kstest(
                event_times,
                "expon",
                args=(0, 1.0 / lambda_mle),
            )
            gof = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_p), 6),
                "exponential_fit_adequate": float(ks_p) > 0.05,
            }
        else:
            gof = {"note": "Too few events for KS goodness-of-fit test."}

        return {
            "lambda_rate": round(lambda_mle, 6),
            "mean_survival_time": round(1.0 / lambda_mle, 4),
            "goodness_of_fit": gof,
        }


class TimeToEventCalculator:
    """Comprehensive time-to-event analysis with threshold and model fitting.

    Workflow:
        calculator = TimeToEventCalculator()
        result = calculator.calculate(
            data_frame=df,
            time_column="days_to_churn",
            event_column="churned",
            thresholds=[30, 60, 90, 180, 365],
            fit_exponential=True,
        )
    """

    _MINIMUM_OBSERVATIONS: int = 5

    def __init__(self) -> None:
        self._summary_calc = TimeToEventSummaryCalculator()
        self._threshold_analyser = ThresholdSurvivalAnalyser()
        self._exp_fitter = ExponentialFitter()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        time_column: str,
        event_column: str,
        thresholds: list[float] | None = None,
        fit_exponential: bool = True,
    ) -> dict:
        """Compute time-to-event descriptive statistics.

        Args:
            data_frame: Source DataFrame.
            time_column: Time-to-event or censoring column.
            event_column: Binary event indicator (1=event, 0=censored).
            thresholds: Time thresholds for survival fraction analysis.
            fit_exponential: Whether to fit an exponential survival model.

        Returns:
            Dict with summary stats, threshold analysis, and optional model fit.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient.
        """
        for col in (time_column, event_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        clean = data_frame[[time_column, event_column]].dropna()
        clean = clean[clean[time_column] >= 0]

        if len(clean) < self._MINIMUM_OBSERVATIONS:
            raise ValueError(
                f"At least {self._MINIMUM_OBSERVATIONS} observations required. "
                f"Got {len(clean)}."
            )

        times = clean[time_column].to_numpy(dtype=float)
        events = clean[event_column].to_numpy(dtype=float)

        summary = self._summary_calc.calculate(times, events)

        effective_thresholds = thresholds if thresholds is not None else [
            round(float(np.percentile(times, p)), 2)
            for p in (25, 50, 75, 90)
        ]

        threshold_analysis = self._threshold_analyser.analyse(times, effective_thresholds)

        exponential_fit = (
            self._exp_fitter.fit(times, events)
            if fit_exponential else None
        )

        return {
            "summary": {
                "n_total": summary.n_total,
                "n_events": summary.n_events,
                "n_censored": summary.n_censored,
                "event_rate": summary.event_rate,
                "mean_time": summary.mean_time,
                "median_time": summary.median_time,
                "std_time": summary.std_time,
                "p25_time": summary.p25_time,
                "p75_time": summary.p75_time,
                "min_time": summary.min_time,
                "max_time": summary.max_time,
                "iqr_time": summary.iqr_time,
            },
            "threshold_analysis": threshold_analysis,
            "exponential_fit": exponential_fit,
            "time_column": time_column,
            "event_column": event_column,
        }