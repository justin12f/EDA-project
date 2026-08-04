"""Estandar docstring idk i'm tired i just dont wanna see the linter"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `TimeSeriesStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
import numpy as np
import pandas as pd

class CenteredMovingAverage:
    """Centered moving average"""

    def calculate(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate the centered moving average"""
        centered_moving_average = data.rolling(window=window, center=True).mean()

        if window % 2 == 0:
            centered_moving_average = centered_moving_average.rolling(2).mean().shift(-1)
        return centered_moving_average

class EstacionalComponent:
    """Class for logic for seasonal component"""

    def calculate(self, data: pd.Series, window: int) -> dict:
        """Calculate the seasonal component"""

        trend = CenteredMovingAverage().calculate(data, window)
        detrended = data - trend
        cycle_index = np.arange(len(detrended)) % window
        seasonal_average = detrended.groupby(cycle_index).mean()
        seasonal_average -= seasonal_average.mean()

        reps = (len(data) // window + 1,)
        seasonal = np.tile(seasonal_average, reps)[: len(data)]
        seasonal = pd.Series(seasonal, index=data.index)

        resid = data - trend - seasonal

        return_dictionary = {
            "trend": trend,
            "detrended": detrended,
            "seasonal_average": seasonal_average,
            "seasonal": seasonal,
            "resid": resid,
        }

        return return_dictionary

class SeasonalDecomposition:
    """Inyeccion dependency for seasonal component"""

    def calculate(self, data: pd.Series, window: int) -> dict:
        """Calculate the seasonal component"""
        return EstacionalComponent().calculate(data, window)
