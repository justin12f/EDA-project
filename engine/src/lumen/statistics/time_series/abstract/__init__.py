"""Abstract contracts for the time series statistics domain."""
from lumen.statistics.time_series.abstract.change_points import AbstractChangePointsCalculator
from lumen.statistics.time_series.abstract.cyclical_patterns import AbstractCyclicalPatternsCalculator
from lumen.statistics.time_series.abstract.forecast_accuracy import AbstractForecastAccuracyCalculator
from lumen.statistics.time_series.abstract.lag_features import AbstractLagFeaturesCalculator
from lumen.statistics.time_series.abstract.momentum import AbstractMomentumCalculator
from lumen.statistics.time_series.abstract.moving_averages import AbstractMovingAveragesCalculator
from lumen.statistics.time_series.abstract.rolling_statistics import AbstractRollingStatisticsCalculator
from lumen.statistics.time_series.abstract.seasonal import AbstractSeasonalCalculator
from lumen.statistics.time_series.abstract.stationarity import AbstractStationarityCalculator
from lumen.statistics.time_series.abstract.volatility import AbstractVolatilityCalculator

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
