"""Abstract contracts for the time series statistics domain."""
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

__all__ = [
    "AbstractChangePointsCalculator",
    "AbstractCyclicalPatternsCalculator",
    "AbstractForecastAccuracyCalculator",
    "AbstractLagFeaturesCalculator",
    "AbstractMomentumCalculator",
    "AbstractMovingAveragesCalculator",
    "AbstractRollingStatisticsCalculator",
    "AbstractSeasonalCalculator",
    "AbstractStationarityCalculator",
    "AbstractVolatilityCalculator",
]
