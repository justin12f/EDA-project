"""Polars-native backend implementations for the time series statistics domain."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

from time_series.abstract.change_points import AbstractChangePointsCalculator
from time_series.abstract.cyclical_patterns import AbstractCyclicalPatternsCalculator
from time_series.abstract.forecast_accuracy import AbstractForecastAccuracyCalculator
from time_series.abstract.lag_features import AbstractLagFeaturesCalculator
from time_series.abstract.momentum import AbstractMomentumCalculator
from time_series.abstract.moving_averages import AbstractMovingAveragesCalculator
from time_series.abstract.rolling_statistics import AbstractRollingStatisticsCalculator
from time_series.abstract.seasonal import AbstractSeasonalCalculator
from time_series.abstract.stationarity import AbstractStationarityCalculator
from time_series.abstract.volatility import AbstractVolatilityCalculator


def _eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


class ChangePointsCalculatorPolars(AbstractChangePointsCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        penalty: float = 1.0,
    ) -> dict[str, Any]:
        import ruptures as rpt
        
        frame = _eager(data).select(pl.col(value_column).drop_nulls())
        arr = frame.to_numpy().flatten()
        
        if len(arr) < 2:
            return {"change_points": []}
            
        algo = rpt.Pelt(model="rbf").fit(arr)
        result = algo.predict(pen=penalty)
        
        # Ruptures includes the end of the array, we remove it
        if result and result[-1] == len(arr):
            result = result[:-1]
            
        return {"change_points": result, "n_change_points": len(result)}


class CyclicalPatternsCalculatorPolars(AbstractCyclicalPatternsCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
    ) -> dict[str, Any]:
        frame = _eager(data).select(pl.col(value_column).drop_nulls())
        arr = frame.to_numpy().flatten()
        
        if len(arr) < 2:
            return {"dominant_frequencies": []}

        # Remove mean
        arr_norm = arr - np.mean(arr)
        
        fft_res = np.fft.rfft(arr_norm)
        freqs = np.fft.rfftfreq(len(arr_norm))
        magnitudes = np.abs(fft_res)
        
        # Ignore freq 0
        freqs = freqs[1:]
        magnitudes = magnitudes[1:]
        
        top_idx = np.argsort(magnitudes)[::-1][:3]
        
        dominant = []
        for idx in top_idx:
            freq = freqs[idx]
            period = 1.0 / freq if freq > 0 else float('inf')
            dominant.append({
                "frequency": float(freq),
                "period": float(period),
                "magnitude": float(magnitudes[idx])
            })

        return {"dominant_frequencies": dominant}


class ForecastAccuracyCalculatorPolars(AbstractForecastAccuracyCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        actual_column: str,
        forecast_column: str,
    ) -> dict[str, Any]:
        frame = _eager(data).select([actual_column, forecast_column]).drop_nulls()
        
        res = frame.select([
            (pl.col(actual_column) - pl.col(forecast_column)).abs().mean().alias("mae"),
            ((pl.col(actual_column) - pl.col(forecast_column))**2).mean().sqrt().alias("rmse"),
            (((pl.col(actual_column) - pl.col(forecast_column)) / pl.col(actual_column)).abs()).mean().alias("mape")
        ]).row(0)

        return {
            "mae": float(res[0]),
            "rmse": float(res[1]),
            "mape": float(res[2]),
        }


class LagFeaturesCalculatorPolars(AbstractLagFeaturesCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        lags: list[int],
    ) -> dict[str, Any]:
        frame = _eager(data)
        
        exprs = []
        for lag in lags:
            exprs.append(pl.col(value_column).shift(lag).alias(f"{value_column}_lag_{lag}"))
            
        res = frame.with_columns(exprs).to_dicts()
        return {"lags": res}


class MomentumCalculatorPolars(AbstractMomentumCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
    ) -> dict[str, Any]:
        # Simple implementation of 14-period RSI
        frame = _eager(data).select([value_column])
        
        frame = frame.with_columns([
            (pl.col(value_column) - pl.col(value_column).shift(1)).alias("change")
        ])
        
        frame = frame.with_columns([
            pl.when(pl.col("change") > 0).then(pl.col("change")).otherwise(0).alias("gain"),
            pl.when(pl.col("change") < 0).then(-pl.col("change")).otherwise(0).alias("loss")
        ])
        
        # Exponential moving average for wilder's smoothing (simplified to rolling mean for Polars)
        frame = frame.with_columns([
            pl.col("gain").rolling_mean(window_size=14, min_periods=1).alias("avg_gain"),
            pl.col("loss").rolling_mean(window_size=14, min_periods=1).alias("avg_loss")
        ])
        
        frame = frame.with_columns([
            (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs")
        ]).with_columns([
            pl.when(pl.col("avg_loss") == 0).then(100).otherwise(100 - (100 / (1 + pl.col("rs")))).alias("rsi_14")
        ])
        
        return {"momentum": frame.to_dicts()}


class MovingAveragesCalculatorPolars(AbstractMovingAveragesCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        windows: list[int],
    ) -> dict[str, Any]:
        frame = _eager(data)
        
        exprs = []
        for w in windows:
            exprs.append(pl.col(value_column).rolling_mean(window_size=w, min_periods=1).alias(f"sma_{w}"))
            # Polars ewm_mean
            exprs.append(pl.col(value_column).ewm_mean(span=w, min_periods=1, ignore_nulls=True).alias(f"ema_{w}"))
            
        res = frame.with_columns(exprs).to_dicts()
        return {"moving_averages": res}


class RollingStatisticsCalculatorPolars(AbstractRollingStatisticsCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        window: int = 14,
    ) -> dict[str, Any]:
        frame = _eager(data)
        
        exprs = [
            pl.col(value_column).rolling_mean(window_size=window, min_periods=1).alias(f"rolling_mean_{window}"),
            pl.col(value_column).rolling_std(window_size=window, min_periods=1).alias(f"rolling_std_{window}"),
            pl.col(value_column).rolling_min(window_size=window, min_periods=1).alias(f"rolling_min_{window}"),
            pl.col(value_column).rolling_max(window_size=window, min_periods=1).alias(f"rolling_max_{window}")
        ]
            
        res = frame.with_columns(exprs).to_dicts()
        return {"rolling_statistics": res}


class SeasonalCalculatorPolars(AbstractSeasonalCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        period: int = 12,
    ) -> dict[str, Any]:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        frame = _eager(data).select(pl.col(value_column).drop_nulls())
        arr = frame.to_numpy().flatten()
        
        if len(arr) < 2 * period:
            return {"error": "Not enough data for seasonal decomposition."}
            
        res = seasonal_decompose(arr, period=period, extrapolate_trend='freq')
        
        return {
            "trend": res.trend.tolist(),
            "seasonal": res.seasonal.tolist(),
            "resid": res.resid.tolist(),
        }


class StationarityCalculatorPolars(AbstractStationarityCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
    ) -> dict[str, Any]:
        frame = _eager(data).select(pl.col(value_column).drop_nulls())
        arr = frame.to_numpy().flatten()
        
        if len(arr) < 10:
            return {"error": "Not enough data"}
            
        res = adfuller(arr)
        
        return {
            "adf_statistic": float(res[0]),
            "p_value": float(res[1]),
            "used_lag": int(res[2]),
            "nobs": int(res[3]),
            "critical_values": {str(k): float(v) for k, v in res[4].items()},
            "is_stationary": float(res[1]) < 0.05
        }


class VolatilityCalculatorPolars(AbstractVolatilityCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        window: int = 30,
    ) -> dict[str, Any]:
        frame = _eager(data)
        
        # Log returns
        frame = frame.with_columns([
            pl.col(value_column).log().diff().alias("log_return")
        ])
        
        # Volatility
        frame = frame.with_columns([
            pl.col("log_return").rolling_std(window_size=window, min_periods=1).alias(f"volatility_{window}")
        ])
        
        return {"volatility": frame.to_dicts()}
