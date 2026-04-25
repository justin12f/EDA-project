"""Module for data analysis"""

from statistics.time_series_analysis.seasonal import SeasonalDecomposition

import pandas as pd

from analyze_data.analyzers.base import BaseDataAnalysis
from models.linear_regression import LinearRegression


class AnalyseDataTypes(BaseDataAnalysis):
    """Analyse the data types of the dataframe"""

    def analyze(self, **kwargs) -> dict[str, str]:
        """Analyse the data type in the columns"""
        data_types: dict[str, str] = self._data_frame.dtypes.to_dict()
        return data_types


class AnalyseDataShape(BaseDataAnalysis):
    """Analyse the shape of the dataframe"""

    def analyze(self, **kwargs) -> tuple[int, int]:
        """Analyse the shape in the columns"""
        return self._data_frame.shape


class AnalyseDataInfo(BaseDataAnalysis):
    """Analyse the info of te data frame"""

    def analyze(self, **kwargs) -> pd.DataFrame:
        """return the summary of the dataframe info"""
        return self._data_frame.info()


class AnalyseDataDescribe(BaseDataAnalysis):
    """Analyse the describe of the dataframe"""

    def analyze(self, **kwargs) -> pd.DataFrame:
        """return the describe  of the data frame"""
        return self._data_frame.describe()


class AnalyseDataColumns(BaseDataAnalysis):
    """Analyse the columns of the dataframe"""

    def analyze(self, **kwargs) -> pd.Index:
        """return the columns of the data frame"""
        return self._data_frame.columns


class AnalyseDataIndex(BaseDataAnalysis):
    """Analyse the index of the dataframe"""

    def analyze(self, **kwargs) -> pd.Index:
        """return the index of the data frame"""
        return self._data_frame.index


class AnalyseDataHead(BaseDataAnalysis):
    """Analyse the head of the dataframe"""

    def analyze(self, **kwargs) -> pd.DataFrame:
        """return the head of the data frame"""
        return self._data_frame.head()


class AnalyseDataTail(BaseDataAnalysis):
    """Analyse the tail of the dataframe"""

    def analyze(self, **kwargs) -> pd.DataFrame:
        """return the tail of the data frame"""
        return self._data_frame.tail()


class AnalyseDataSample(BaseDataAnalysis):
    """Analyse the sample of the dataframe"""

    def analyze(self, **kwargs) -> pd.DataFrame:
        """return the sample of the data frame"""
        return self._data_frame.sample()


# ==================== ANALYZERS DE FEATURES ENGINEERING ====================


# TENDENCIAS Y PATRONES
# ("trend_analysis", AnalyseTrendPatterns)      # Tendencia temporal


class AnalyseSeasonality(BaseDataAnalysis):
    """Analyse the seasonality of the data frame"""

    def analyze(self, **kwargs) -> dict:
        """return the seasonality of the data frame"""

        data_frame: pd.DataFrame = self._data_frame
        target_column: str = kwargs.get("target_column")
        period: int = kwargs.get("period")
        if period is None:
            period = 3

        if data_frame is None or target_column is None:
            raise ValueError("data_frame and target_column must be provided")

        # Relaxed index validation: we assume the data is already sorted sequentially.
        # if not isinstance(data_frame.index, pd.DatetimeIndex):
        #     raise ValueError(
        #         "The index of the data frame must be of type DatetimeIndex"
        #     )

        if target_column not in data_frame.columns:
            raise KeyError(
                f"The column '{target_column}' does not exist in the data frame"
            )

        seasonal_decomposition = SeasonalDecomposition().calculate(
            data_frame[target_column], window=period
        )

        return {
            "period_detected": period,
            "components": {
                "observed": data_frame[target_column],
                "trend": seasonal_decomposition["trend"],
                "seasonal": seasonal_decomposition["seasonal"],
                "resid": seasonal_decomposition["resid"],
            },
        }


# ("volatility", AnalyseVolatility)             # Volatilidad en series

# ("momentum", AnalyseMomentum)                 # Cambios acelerados

# RELACIONES Y DEPENDENCIAS
# ("correlation", AnalyseCorrelation)           # Correlaciones lineales
# ("causality", AnalyseCausality)               # Relaciones causa-efecto
# ("interaction_effects", AnalyseInteractions)  # Variables que se potencian
# ("multicollinearity", AnalyseMulticollinearity) # VIF, redundancias

# DISTRIBUCIONES (PARA TRANSFORMACIONES)
# ("distribution_type", AnalyseDistributionType) # Normal, exponencial, etc
# ("skewness_kurtosis", AnalyseSkewnessKurtosis) # Para normalizar
# ("normality_test", AnalyseNormalityTests)     # Shapiro, Anderson-Darling

# INDICADORES FINANCIEROS / ECONÓMICOS
# ("financial_ratios", AnalyseFinancialRatios)  # ROE, ROA, etc
# ("risk_metrics", AnalyseRiskMetrics)          # VaR, sharpe ratio
# ("growth_rates", AnalyseGrowthRates)          # YoY, MoM
# ("moving_averages", AnalyseMovingAverages)    # MA50, MA200

# DETECTAR ANOMALÍAS (PARA MODELOS)
# ("anomaly_scores", AnalyseAnomalyScores)      # Isolation Forest, LOF
# ("change_points", AnalyseChangePoints)        # CUSUM, Pelt
# ("threshold_violations", AnalyseThresholds)   # % fuera de límites

# AGREGACIONES Y PIVOTS (PARA FEATURES)
# ("group_statistics", AnalyseGroupStats)       # Agg por categoría
# ("rolling_features", AnalyseRollingFeatures)  # Media móvil, std móvil
# ("lag_features", AnalyseLagFeatures)          # t-1, t-2, t-n
# ("diff_features", AnalyseDiffFeatures)        # Diferencias periodo a periodo

# SEGMENTACIÓN Y CLUSTERS
# ("customer_segments", AnalyseSegmentation)    # RFM, K-means clusters
# ("cohort_analysis", AnalyseCohortAnalysis)    # Análisis por cohortes
# ("population_splits", AnalysePopulationSplits) # A/B ready

# FEATURE IMPORTANCE (PREVIA A ML)
# ("feature_variance", AnalyseFeatureVariance)  # Qué variables varían más
# ("feature_selection", AnalyseFeatureSelection) # Mutual info, chi2
# ("information_content", AnalyseInformationContent) # Entropy, MI


class AnalyseTrendPatterns(BaseDataAnalysis):
    """Analyse the trend patterns of the dataframe"
    Work Flow:
    1. Create a data frame
    trend_analyzer = AnalyzerFactory.create("pd.DataFrame" , result)

    2. Analyze the data frame
    result_trend = trend_analyzer.analyze(
        x="column_name",
        y="column_name",
        type_of_analysis="gradient_descent" or "ordinary_least_squares",
        complexity="simple" or "multiple"        )

    """

    def analyze(self, **kwargs) -> dict:
        """return the trend patterns of the data frame"""

        x_name = kwargs.get("x")
        y_name = kwargs.get("y")

        if x_name is None or y_name is None:
            raise ValueError("x and y must be provided")

        x = self._data_frame[x_name]
        y = self._data_frame[y_name]

        # Strip analyzer-level keys so they don't conflict with fit() parameters
        fit_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in ("x", "y", "type_of_prediction", "type_of_analysis", "complexity")
        }

        # Accept type_of_analysis or type_of_prediction
        prediction_type = kwargs.get(
            "type_of_analysis",
            kwargs.get("type_of_prediction", "ordinary_least_squares"),
        )

        linear_regression_arguments = {
            "type_of_prediction": prediction_type,
            "complexity": kwargs.get("complexity", "simple"),
        }

        linear_regression_object = LinearRegression(**linear_regression_arguments)
        linear_regression_object.fit(x, y, **fit_kwargs)
        linear_regression_object.predict(x)

        inner_model = linear_regression_object.model

        if hasattr(inner_model, "intercept_"):
            intercept = inner_model.intercept_
            slope = inner_model.slope_
        else:
            # For gradient descent or multiple linear regression that uses design matrix
            intercept = inner_model.coefficients_[0]
            slope = inner_model.coefficients_[1]

        return_dictionary = {
            "trend_slope": slope,
            "trend_intercept": intercept,
            "score": linear_regression_object.score(y),
        }

        return return_dictionary
