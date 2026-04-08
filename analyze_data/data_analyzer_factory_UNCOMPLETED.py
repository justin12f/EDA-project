"""Module for data analysis"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseDataAnalysis(ABC):
    """base class for data analysis"""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        self._data_frame = data_frame

    @abstractmethod
    def analyze(self) -> any:
        """analyze the data frame and return the results"""


class AnalyseDataTypes(BaseDataAnalysis):
    """Analyse the data types of the dataframe"""

    @abstractmethod
    def analyze(self) -> dict:
        """Analyse the data type in the columns"""
        data_types: dict[tuple[str, str]] = {}
        for column in self._data_frame.columns:
            information = (column, self._data_frame[column].dtype)
            data_types.update(information)
        return data_types


class AnalyseDataShape(BaseDataAnalysis):
    """Analyse the shape of the dataframe"""

    def analyse(self) -> tuple[int, int]:
        """Analyse the shape in the columns"""
        return self._data_frame.shape


class AnalyseDataInfo(BaseDataAnalysis):
    """Analyse the info of te data frame"""

    def analyse(self) -> pd.DataFrame.info:
        """return the summary of the dataframe info"""
        return self._data_frame.info


class AnalyseDataDescribe(BaseDataAnalysis):
    """Analyse the describe of the dataframe"""

    def analyse(self) -> pd.DataFrame.describe:
        """return the describe  of the data frame"""
        return self._data_frame.describe


class AnalyseDataColumns(BaseDataAnalysis):
    """Analyse the columns of the dataframe"""

    def analyse(self) -> pd.DataFrame.columns:
        """return the columns of the data frame"""
        return self._data_frame.columns


class AnalyseDataIndex(BaseDataAnalysis):
    """Analyse the index of the dataframe"""

    def analyse(self) -> pd.DataFrame.index:
        """return the index of the data frame"""
        return self._data_frame.index


class AnalyseDataHead(BaseDataAnalysis):
    """Analyse the head of the dataframe"""

    def analyse(self) -> pd.DataFrame.head:
        """return the head of the data frame"""
        return self._data_frame.head


class AnalyseDataTail(BaseDataAnalysis):
    """Analyse the tail of the dataframe"""

    def analyse(self) -> pd.DataFrame.tail:
        """return the tail of the data frame"""
        return self._data_frame.tail


class AnalyseDataSample(BaseDataAnalysis):
    """Analyse the sample of the dataframe"""

    def analyse(self) -> pd.DataFrame.sample:
        """return the sample of the data frame"""
        return self._data_frame.sample

# ==================== ANALYZERS DE FEATURES ENGINEERING ====================

# TENDENCIAS Y PATRONES
#("trend_analysis", AnalyseTrendPatterns)      # Tendencia temporal
class AnalyseTrendPatterns(BaseDataAnalysis):
    """Analyse the trend patterns of the dataframe"""
    def analyse ( self ) :
        """return the trend patterns of the data frame"""
#("seasonality", AnalyseSeasonality)           # Patrones estacionales
#("volatility", AnalyseVolatility)             # Volatilidad en series
#("momentum", AnalyseMomentum)                 # Cambios acelerados

# RELACIONES Y DEPENDENCIAS
#("correlation", AnalyseCorrelation)           # Correlaciones lineales
#("causality", AnalyseCausality)               # Relaciones causa-efecto
#("interaction_effects", AnalyseInteractions)  # Variables que se potencian
#("multicollinearity", AnalyseMulticollinearity) # VIF, redundancias

# DISTRIBUCIONES (PARA TRANSFORMACIONES)
#("distribution_type", AnalyseDistributionType) # Normal, exponencial, etc
#("skewness_kurtosis", AnalyseSkewnessKurtosis) # Para normalizar
#("normality_test", AnalyseNormalityTests)     # Shapiro, Anderson-Darling

# INDICADORES FINANCIEROS / ECONÓMICOS
#("financial_ratios", AnalyseFinancialRatios)  # ROE, ROA, etc
#("risk_metrics", AnalyseRiskMetrics)          # VaR, sharpe ratio
#("growth_rates", AnalyseGrowthRates)          # YoY, MoM
#("moving_averages", AnalyseMovingAverages)    # MA50, MA200

# DETECTAR ANOMALÍAS (PARA MODELOS)
#("anomaly_scores", AnalyseAnomalyScores)      # Isolation Forest, LOF
#("change_points", AnalyseChangePoints)        # CUSUM, Pelt
#("threshold_violations", AnalyseThresholds)   # % fuera de límites

# AGREGACIONES Y PIVOTS (PARA FEATURES)
#("group_statistics", AnalyseGroupStats)       # Agg por categoría
#("rolling_features", AnalyseRollingFeatures)  # Media móvil, std móvil
#("lag_features", AnalyseLagFeatures)          # t-1, t-2, t-n
#("diff_features", AnalyseDiffFeatures)        # Diferencias periodo a periodo

# SEGMENTACIÓN Y CLUSTERS
#("customer_segments", AnalyseSegmentation)    # RFM, K-means clusters
#("cohort_analysis", AnalyseCohortAnalysis)    # Análisis por cohortes
#("population_splits", AnalysePopulationSplits) # A/B ready

# FEATURE IMPORTANCE (PREVIA A ML)
#("feature_variance", AnalyseFeatureVariance)  # Qué variables varían más
#("feature_selection", AnalyseFeatureSelection) # Mutual info, chi2
#("information_content", AnalyseInformationContent) # Entropy, MI


class AnalyzerFactory:
    """Factory for creating data analyzer


    Usage:
    1. Register an analyzer: AnalizerFactory.register("name", AnalyzerClass)
    2. Create an analyzer:   AnalizerFactory.create("name", data_frame)
    3. Analyze the data:     AnalizerFactory.create("name", data_frame).analyze()
    """

    _registry: dict[str, type[BaseDataAnalysis]] = {}

    @classmethod
    def register(cls, name: str, analyzer: type[BaseDataAnalysis]) -> None:
        """Register a new analyzer"""
        cls._registry[name] = analyzer

    @classmethod
    def create(cls, name: str, data_frame: pd.DataFrame) -> BaseDataAnalysis:
        """Create a new analyzer"""
        analyzer = cls._registry.get(name)
        if not analyzer:
            raise ValueError(f"Analyzer {name} not registered")
        return analyzer(data_frame)




AnalyzerFactory.register("types", AnalyseDataTypes)
AnalyzerFactory.register("shape", AnalyseDataShape)
AnalyzerFactory.register("info", AnalyseDataInfo)
AnalyzerFactory.register("describe", AnalyseDataDescribe)
AnalyzerFactory.register("columns", AnalyseDataColumns)
AnalyzerFactory.register("index", AnalyseDataIndex)
AnalyzerFactory.register("head", AnalyseDataHead)
AnalyzerFactory.register("tail", AnalyseDataTail)
AnalyzerFactory.register("sample", AnalyseDataSample)
