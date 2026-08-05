"""Time-series forecasting.

Implemented natively in numpy rather than through statsmodels. Holt-Winters is
about forty lines of recurrence; taking a ~30MB dependency and its build chain to
avoid writing them is a bad trade for a container that ships to production, and
the engine's stated rule is native backend APIs.

Every method here treats row order as time order. Nothing checks that it is —
that is the caller's claim, and `extract.py` is where the sort happens.
"""

from __future__ import annotations

import numpy as np

from lumen.prediction.base import (
    Family,
    FitReport,
    Prediction,
    Predictor,
    Task,
    check_training_data,
)


class NaiveForecast(Predictor):
    """Tomorrow equals today.

    Trivial, and the benchmark that matters: on many real series nothing beats
    it, and a method that loses to it has told you something important about
    itself.
    """

    name = "naive"
    family = Family.TIMESERIES
    task = Task.FORECAST
    univariate = True

    def __init__(self) -> None:
        super().__init__()
        self._last: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        _, y = check_training_data(X, y, minimum=1)
        self._last = float(y[-1])
        self._fitted = True
        return FitReport(
            predictor=self.name, family=self.family, task=self.task,
            n_samples=y.size, n_features=0, params={},
            diagnostics={"last_value": self._last},
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        steps = np.asarray(X).reshape(-1).size
        return Prediction(values=np.full(steps, self._last, dtype=float))


class SeasonalNaive(Predictor):
    """Repeat the last observed season."""

    name = "seasonal_naive"
    family = Family.TIMESERIES
    task = Task.FORECAST
    univariate = True

    def __init__(self, season_length: int = 7) -> None:
        if season_length < 2:
            raise ValueError("season_length must be at least 2")
        super().__init__(season_length=season_length)
        self._season: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        season = self.params["season_length"]
        _, y = check_training_data(X, y, minimum=season)
        self._season = y[-season:].copy()
        self._fitted = True
        return FitReport(
            predictor=self.name, family=self.family, task=self.task,
            n_samples=y.size, n_features=0, params=dict(self.params),
            diagnostics={"season": [float(v) for v in self._season]},
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        steps = np.asarray(X).reshape(-1).size
        repeats = int(np.ceil(steps / self._season.size))
        return Prediction(values=np.tile(self._season, repeats)[:steps])


class DriftForecast(Predictor):
    """Extend the straight line between the first and last observation.

    Deliberately not a least-squares trend: drift is the standard benchmark and
    uses only the endpoints, which makes it robust to the shape in between and
    trivial to explain to whoever has to approve the forecast.
    """

    name = "drift"
    family = Family.TIMESERIES
    task = Task.FORECAST
    univariate = True

    def __init__(self) -> None:
        super().__init__()
        self._last: float | None = None
        self._slope: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        _, y = check_training_data(X, y, minimum=2)
        self._last = float(y[-1])
        self._slope = float((y[-1] - y[0]) / (y.size - 1))
        self._fitted = True
        return FitReport(
            predictor=self.name, family=self.family, task=self.task,
            n_samples=y.size, n_features=0, params={},
            diagnostics={"slope_per_step": self._slope, "last_value": self._last},
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        steps = np.asarray(X).reshape(-1).size
        ahead = np.arange(1, steps + 1, dtype=float)
        return Prediction(values=self._last + self._slope * ahead)


class ExponentialSmoothing(Predictor):
    """Holt-Winters, with optional trend and seasonality.

    Three configurations from one recurrence:
      simple  level only                       — no trend, no season
      Holt    level + trend                    — trend=True
      Holt-Winters  level + trend + season     — trend=True, season_length set

    Additive seasonality only. Multiplicative needs a strictly positive series,
    and silently switching on the data's sign is the kind of hidden branch that
    makes a forecast unexplainable.
    """

    name = "exponential_smoothing"
    family = Family.TIMESERIES
    task = Task.FORECAST
    univariate = True

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.1,
        trend: bool = True,
        season_length: int | None = None,
    ) -> None:
        for label, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{label} must be in (0, 1); got {value}")
        super().__init__(
            alpha=alpha, beta=beta, gamma=gamma, trend=trend, season_length=season_length
        )
        self._level: float | None = None
        self._trend: float = 0.0
        self._season: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitReport:
        season_length = self.params["season_length"]
        minimum = 2 * season_length if season_length else 2
        _, y = check_training_data(X, y, minimum=minimum)

        alpha, beta, gamma = (self.params[k] for k in ("alpha", "beta", "gamma"))
        use_trend = bool(self.params["trend"])

        if season_length:
            # Seed the seasonal figures from the mean of each position across
            # whole cycles — the standard initialisation, and stable enough that
            # the first cycle does not dominate.
            cycles = y.size // season_length
            grid = y[: cycles * season_length].reshape(cycles, season_length)
            season = grid.mean(axis=0) - grid.mean()
            level = float(grid[0].mean())
        else:
            season = None
            level = float(y[0])

        trend = float(y[1] - y[0]) if use_trend and y.size > 1 else 0.0

        for index, observation in enumerate(y):
            seasonal = season[index % season_length] if season is not None else 0.0
            previous_level = level

            level = alpha * (observation - seasonal) + (1 - alpha) * (level + trend)
            if use_trend:
                trend = beta * (level - previous_level) + (1 - beta) * trend
            if season is not None:
                season[index % season_length] = (
                    gamma * (observation - level) + (1 - gamma) * seasonal
                )

        self._level, self._trend, self._season = level, trend, season
        self._n = y.size
        self._fitted = True

        return FitReport(
            predictor=self.name, family=self.family, task=self.task,
            n_samples=y.size, n_features=0, params=dict(self.params),
            diagnostics={
                "level": float(level),
                "trend_per_step": float(trend),
                "seasonal": [float(v) for v in season] if season is not None else None,
                "configuration": (
                    "holt-winters" if season is not None
                    else "holt" if use_trend
                    else "simple"
                ),
            },
        )

    def predict(self, X: np.ndarray) -> Prediction:
        self._require_fitted()
        steps = np.asarray(X).reshape(-1).size
        ahead = np.arange(1, steps + 1, dtype=float)

        values = self._level + (self._trend * ahead if self.params["trend"] else 0.0)
        if self._season is not None:
            length = self._season.size
            offsets = np.array(
                [self._season[(self._n + step - 1) % length] for step in range(1, steps + 1)]
            )
            values = values + offsets
        return Prediction(values=np.asarray(values, dtype=float))


def detect_season(y: np.ndarray, max_period: int | None = None) -> int | None:
    """Find the dominant seasonal period by autocorrelation, or None.

    A caller who must supply `season_length` already knows it; an agent looking
    at an unfamiliar column does not. Without detection a seasonal method runs
    with no season and is wrong by roughly the seasonal amplitude — which reads
    as a mediocre forecast rather than as a missing argument.

    Two things make this harder than "argmax of the autocorrelation":

    * A trend dominates every lag and buries the periodic signal, so the series
      is detrended first.
    * On any smooth series adjacent points are similar, so short lags score
      highest whatever the real period is. A sine with a 30-step cycle has
      ~0.99 correlation at lag 2. The period is therefore taken as the first
      local *peak* — a lag that beats both its neighbours — not the global
      maximum.

    Returns None when nothing qualifies. A wrong period is worse than none: it
    imposes a rhythm the data does not have.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if y.size < 8:
        return None

    # Two full cycles are the minimum that distinguishes a season from a
    # coincidence, so no candidate may exceed half the series.
    ceiling = min(max_period or y.size // 2, y.size // 2)
    if ceiling < 2:
        return None

    steps = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(steps, y, 1)
    residual = y - (slope * steps + intercept)

    variance = float(np.dot(residual, residual))
    scale = float(np.dot(y - y.mean(), y - y.mean()))
    # A pure trend leaves only floating-point dust behind, which then correlates
    # perfectly with itself. Anything under 0.1% of the original variation is
    # noise, not a season.
    if variance == 0.0 or scale == 0.0 or variance / scale < 1e-3:
        return None

    correlations = np.array(
        [
            float(np.dot(residual[:-lag], residual[lag:]) / variance)
            for lag in range(1, ceiling + 1)
        ]
    )

    # 0.3 is a judgment call: high enough that noise does not qualify, low
    # enough that a real but noisy weekly rhythm still does.
    threshold = 0.3
    for index in range(1, correlations.size - 1):
        lag = index + 1
        if (
            correlations[index] >= threshold
            and correlations[index] > correlations[index - 1]
            and correlations[index] >= correlations[index + 1]
        ):
            return lag
    return None
