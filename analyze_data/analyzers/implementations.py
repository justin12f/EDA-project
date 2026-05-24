"""Module for data analysis"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Verificar si es capa abstracta/legacy; registrar solo contratos en factory maestra y delegar implementaciones a `analyze_data/analyzers/backends/`.
# - ABSTRACCIÓN DEL DATO: Contratos sin tipos pandas fijos; usar TypeVar del contenedor por backend.
# - REFACTOR NATIVO: Eliminar archivo si está obsoleto y sin referencias; si se conserva, solo ABC + registro sin lógica NumPy/Pandas.
# #[AI_CONTEXT_END]
from statistics.time_series.seasonal import SeasonalDecomposition

import pandas as pd

from analyze_data.analyzers.base import BaseDataAnalysis
from models.linear_regression import LinearRegression
import numpy as np
from statistics.descriptive.distribution import DistributionClassifier
from statistics.descriptive.normality import NormalityTestSuite
from statistics.descriptive.central_tendency import CentralTendencyCalculator
from statistics.descriptive.dispersion import DispersionCalculator
from statistics.descriptive.frequency import FrequencyDistributionBuilder
from statistics.descriptive.percentiles import PercentilesCalculator
from statistics.descriptive.skewness_kurtosis import SkewnessKurtosisCalculator
from statistics.descriptive.value_counts import ValueCountsCalculator

# ── DOMAIN 2 IMPORTS ──────────────────────────────────────────────────────────
"""Module for data analysis"""

from statistics.descriptive.distribution import DistributionClassifier
from statistics.descriptive.normality import NormalityTestSuite
from statistics.descriptive.central_tendency import CentralTendencyCalculator
from statistics.descriptive.dispersion import DispersionCalculator
from statistics.descriptive.frequency import FrequencyDistributionBuilder
from statistics.descriptive.percentiles import PercentilesCalculator
from statistics.descriptive.skewness_kurtosis import SkewnessKurtosisCalculator
from statistics.descriptive.value_counts import ValueCountsCalculator

# ── DOMAIN 2 IMPORTS ──────────────────────────────────────────────────────────
from statistics.inferential.hypothesis_test import HypothesisTestSuite
from statistics.inferential.anova import OneWayAnovaCalculator
from statistics.inferential.chi_square import ChiSquareTestCalculator
from statistics.inferential.correlation_significance import CorrelationSignificanceCalculator
from statistics.inferential.confidence_intervals import ConfidenceIntervalCalculator
from statistics.inferential.effect_size import EffectSizeCalculator
from statistics.inferential.power_analysis import PowerAnalysisCalculator
from statistics.inferential.bootstrap import BootstrapEstimator

# ── DOMAIN 3 IMPORTS ──────────────────────────────────────────────────────────
from statistics.relational.correlation_matrix import CorrelationMatrixCalculator
from statistics.relational.multicollinearity import MulticollinearityCalculator
from statistics.relational.mutual_information import MutualInformationCalculator
from statistics.relational.partial_correlation import PartialCorrelationCalculator
from statistics.relational.cross_correlation import CrossCorrelationCalculator
from statistics.relational.granger_causality import GrangerCausalityCalculator
from statistics.relational.contingency_analysis import ContingencyAnalysisCalculator
from statistics.relational.interaction_effects import InteractionEffectsCalculator

# ── DOMAIN 6 IMPORTS ──────────────────────────────────────────────────────────
from statistics.nlp.text_basic_stats import TextBasicStatsCalculator
from statistics.nlp.word_frequency import WordFrequencyCalculator
from statistics.nlp.sentiment_analysis import SentimentAnalysisCalculator
from statistics.nlp.topic_detection import TopicDetectionCalculator
from statistics.nlp.language_detection import LanguageDetectionCalculator
from statistics.nlp.text_similarity import TextSimilarityCalculator
from statistics.nlp.named_entity_density import NamedEntityDensityCalculator

# ── DOMAIN 9 IMPORTS ──────────────────────────────────────────────────────────
from statistics.geospatial.geo_distribution import GeoDistributionCalculator
from statistics.geospatial.geo_clustering import GeoClusteringCalculator
from statistics.geospatial.geo_bounding_box import GeoBoundingBoxCalculator
from statistics.geospatial.geo_heatmap import GeoHeatmapCalculator
from statistics.geospatial.proximity_analysis import ProximityAnalysisCalculator
# ── DOMAIN 7 IMPORTS ──────────────────────────────────────────────────────────
from statistics.segmentation.kmeans_clusters import KMeansClusterCalculator
from statistics.segmentation.rfm_segmentation import RFMSegmentationCalculator
from statistics.segmentation.cohort_analysis import CohortAnalysisCalculator
from statistics.segmentation.population_splits import PopulationSplitsCalculator
from statistics.segmentation.dbscan_clusters import DBSCANClusterCalculator
from statistics.segmentation.hierarchical_clusters import HierarchicalClusterCalculator

# ── DOMAIN 8 IMPORTS ──────────────────────────────────────────────────────────
from statistics.business.growth_rates import GrowthRatesCalculator
from statistics.business.risk_metrics import RiskMetricsCalculator
from statistics.business.financial_ratios import FinancialRatiosCalculator
from statistics.business.conversion_funnel import ConversionFunnelCalculator
from statistics.business.churn_rate import ChurnRateCalculator
from statistics.business.customer_lifetime_value import CustomerLifetimeValueCalculator
from statistics.business.pareto_analysis import ParetoAnalysisCalculator
from statistics.business.run_rate import RunRateCalculator

# ── DOMAIN 10 IMPORTS ─────────────────────────────────────────────────────────
from statistics.graphs.network_density import NetworkDensityCalculator
from statistics.graphs.centrality_analysis import CentralityAnalysisCalculator
from statistics.graphs.community_detection import CommunityDetectionCalculator
from statistics.graphs.path_analysis import PathAnalysisCalculator

# ── DOMAIN 11 IMPORTS ─────────────────────────────────────────────────────────
from statistics.survival.kaplan_meier import KaplanMeierCalculator
from statistics.survival.hazard_rate import HazardRateCalculator
from statistics.survival.event_density import EventDensityCalculator
from statistics.survival.time_to_event import TimeToEventCalculator

# ── BASIC DATAFRAME INSPECTORS ────────────────────────────────────────────────

class AnalyseDataTypes(BaseDataAnalysis):
    """Return a dict of column → dtype string for the DataFrame."""

    def analyze(self, **kwargs) -> dict:
        return {"dtypes": self._data_frame.dtypes.astype(str).to_dict()}

class AnalyseDataShape(BaseDataAnalysis):
    """Return the (rows, columns) shape of the DataFrame."""

    def analyze(self, **kwargs) -> dict:
        return {"rows": self._data_frame.shape[0], "columns": self._data_frame.shape[1]}

class AnalyseDataInfo(BaseDataAnalysis):
    """Return column names, dtypes, and non-null counts."""

    def analyze(self, **kwargs) -> dict:
        info = {
            col: {
                "dtype": str(self._data_frame[col].dtype),
                "non_null": int(self._data_frame[col].notna().sum()),
                "null": int(self._data_frame[col].isna().sum()),
            }
            for col in self._data_frame.columns
        }
        return {"info": info, "n_rows": len(self._data_frame)}

class AnalyseDataDescribe(BaseDataAnalysis):
    """Return pandas describe() as a nested dict."""

    def analyze(self, **kwargs) -> dict:
        include = kwargs.get("include", "all")
        return self._data_frame.describe(include=include).to_dict()

class AnalyseDataColumns(BaseDataAnalysis):
    """Return the list of column names."""

    def analyze(self, **kwargs) -> dict:
        return {"columns": self._data_frame.columns.tolist()}

class AnalyseDataIndex(BaseDataAnalysis):
    """Return index information."""

    def analyze(self, **kwargs) -> dict:
        idx = self._data_frame.index
        return {
            "index_name": idx.name,
            "index_dtype": str(idx.dtype),
            "start": str(idx[0]) if len(idx) > 0 else None,
            "end": str(idx[-1]) if len(idx) > 0 else None,
            "n": len(idx),
        }

class AnalyseDataHead(BaseDataAnalysis):
    """Return the first N rows as a dict."""

    def analyze(self, **kwargs) -> dict:
        n: int = kwargs.get("n", 5)
        return self._data_frame.head(n).to_dict(orient="records")

class AnalyseDataTail(BaseDataAnalysis):
    """Return the last N rows as a dict."""

    def analyze(self, **kwargs) -> dict:
        n: int = kwargs.get("n", 5)
        return self._data_frame.tail(n).to_dict(orient="records")

class AnalyseDataSample(BaseDataAnalysis):
    """Return a random sample of N rows as a dict."""

    def analyze(self, **kwargs) -> dict:
        n: int = kwargs.get("n", 5)
        random_state: int = kwargs.get("random_state", 42)
        sample_n = min(n, len(self._data_frame))
        return self._data_frame.sample(sample_n, random_state=random_state).to_dict(
            orient="records"
        )

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

# ── DOMAIN 1 — DESCRIPTIVE STATISTICS ─────────────────────────────────────────

class AnalyseDistributionType(BaseDataAnalysis):
    """Classify the statistical distribution of a numerical column.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="price",
            significance_level=0.05,  # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' parameter is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")

        data: np.ndarray = self._data_frame[column].dropna().to_numpy()
        return DistributionClassifier().classify(data)

class AnalyseSkewnessKurtosis(BaseDataAnalysis):
    """Analyse skewness and kurtosis across numerical columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            columns=["price", "quantity"],  # optional, defaults to all numeric
            bias=True,                      # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        columns: list[str] | None = kwargs.get("columns")
        bias: bool = kwargs.get("bias", True)

        numeric_df = self._data_frame.select_dtypes(include=[np.number])

        if columns is not None:
            missing = [c for c in columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {missing}")
            numeric_df = numeric_df[columns]

        if numeric_df.empty:
            raise ValueError("No numeric columns found in the DataFrame.")

        calculator = SkewnessKurtosisCalculator()
        column_results = {
            col: calculator.calculate(numeric_df[col].dropna().to_numpy(), bias=bias)
            for col in numeric_df.columns
            if len(numeric_df[col].dropna()) >= 4
        }

        return {"columns": column_results, "bias": bias}

class AnalyseNormalityTests(BaseDataAnalysis):
    """Run a full normality test suite on a numerical column.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="price",
            significance_level=0.05,  # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        significance_level: float = kwargs.get("significance_level", 0.05)

        if column is None:
            raise ValueError("'column' parameter is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")

        data: np.ndarray = self._data_frame[column].dropna().to_numpy()
        return NormalityTestSuite().run(data, significance_level=significance_level)

class AnalyseValueCounts(BaseDataAnalysis):
    """Analyse value frequencies for any column (numeric or categorical).

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="category",
            top_n=20,              # optional
            include_missing=True,  # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        top_n: int | None = kwargs.get("top_n")
        include_missing: bool = kwargs.get("include_missing", True)

        if column is None:
            raise ValueError("'column' parameter is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")

        return ValueCountsCalculator().calculate(
            self._data_frame[column],
            top_n=top_n,
            include_missing=include_missing,
        )

class AnalysePercentiles(BaseDataAnalysis):
    """Analyse percentile distribution of a numerical column.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="price",
            percentiles=[5, 25, 50, 75, 95],  # optional
            outlier_bounds=(1, 99),            # optional, None to skip
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        percentiles: list[int] | None = kwargs.get("percentiles")
        outlier_bounds: tuple[int, int] | None = kwargs.get("outlier_bounds", (1, 99))

        if column is None:
            raise ValueError("'column' parameter is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")

        data: np.ndarray = self._data_frame[column].dropna().to_numpy()
        return PercentilesCalculator().calculate(
            data, percentiles=percentiles, outlier_bounds=outlier_bounds
        )

class AnalyseFrequencyDistribution(BaseDataAnalysis):
    """Analyse frequency distribution (histogram as table) of a numerical column.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="price",
            n_bins=15,             # optional, auto-selected if omitted
            bin_method="sturges",  # optional: "sturges", "scott", "fd", "auto"
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        n_bins: int | None = kwargs.get("n_bins")
        bin_method: str = kwargs.get("bin_method", "auto")

        if column is None:
            raise ValueError("'column' parameter is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' does not exist in the DataFrame.")

        data: np.ndarray = self._data_frame[column].dropna().to_numpy()
        return FrequencyDistributionBuilder().build(
            data, n_bins=n_bins, bin_method=bin_method
        )

class AnalyseCentralTendency(BaseDataAnalysis):
    """Analyse central tendency measures across numerical columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            columns=["price", "quantity"],  # optional, defaults to all numeric
            trim_proportion=0.1,            # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        columns: list[str] | None = kwargs.get("columns")
        trim_proportion: float = kwargs.get("trim_proportion", 0.1)

        numeric_df = self._data_frame.select_dtypes(include=[np.number])

        if columns is not None:
            missing = [c for c in columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {missing}")
            numeric_df = numeric_df[columns]

        if numeric_df.empty:
            raise ValueError("No numeric columns found in the DataFrame.")

        calculator = CentralTendencyCalculator()
        column_results = {
            col: calculator.calculate(
                numeric_df[col].dropna().to_numpy(),
                trim_proportion=trim_proportion,
            )
            for col in numeric_df.columns
            if len(numeric_df[col].dropna()) > 0
        }

        return {"columns": column_results, "trim_proportion": trim_proportion}

class AnalyseDispersion(BaseDataAnalysis):
    """Analyse dispersion measures across numerical columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            columns=["price", "quantity"],  # optional, defaults to all numeric
            ddof=1,                         # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        columns: list[str] | None = kwargs.get("columns")
        ddof: int = kwargs.get("ddof", 1)

        numeric_df = self._data_frame.select_dtypes(include=[np.number])

        if columns is not None:
            missing = [c for c in columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {missing}")
            numeric_df = numeric_df[columns]

        if numeric_df.empty:
            raise ValueError("No numeric columns found in the DataFrame.")

        calculator = DispersionCalculator()
        column_results = {
            col: calculator.calculate(numeric_df[col].dropna().to_numpy(), ddof=ddof)
            for col in numeric_df.columns
            if len(numeric_df[col].dropna()) > 0
        }

        return {"columns": column_results, "ddof": ddof}

class AnalyseHypothesisTest(BaseDataAnalysis):
    """Run a parametric or non-parametric hypothesis test on two DataFrame columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_a="sales_control",
            column_b="sales_treatment",   # omit for one-sample t-test
            test="t_test",                # "t_test" | "mann_whitney" | "wilcoxon"
            significance_level=0.05,
            alternative="two-sided",      # "two-sided" | "less" | "greater"
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_a: str = kwargs.get("column_a")
        column_b: str | None = kwargs.get("column_b")

        if column_a is None:
            raise ValueError("'column_a' is required.")
        if column_a not in self._data_frame.columns:
            raise KeyError(f"Column '{column_a}' not found in DataFrame.")
        if column_b is not None and column_b not in self._data_frame.columns:
            raise KeyError(f"Column '{column_b}' not found in DataFrame.")

        group_a = self._data_frame[column_a].dropna().to_numpy()
        group_b = (
            self._data_frame[column_b].dropna().to_numpy()
            if column_b is not None
            else None
        )

        return HypothesisTestSuite().run(
            group_a=group_a,
            group_b=group_b,
            test=kwargs.get("test", "t_test"),
            significance_level=kwargs.get("significance_level", 0.05),
            alternative=kwargs.get("alternative", "two-sided"),
        )

class AnalyseAnova(BaseDataAnalysis):
    """Run one-way ANOVA across multiple DataFrame columns as groups.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            groups={"control": "col_a", "treatment_1": "col_b", "treatment_2": "col_c"},
            significance_level=0.05,
            run_post_hoc=True,
        )
    """

    def analyze(self, **kwargs) -> dict:
        group_column_map: dict[str, str] = kwargs.get("groups")

        if group_column_map is None:
            raise ValueError(
                "'groups' dict mapping group names to column names is required."
            )

        missing_cols = [
            col for col in group_column_map.values()
            if col not in self._data_frame.columns
        ]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

        groups: dict[str, np.ndarray] = {
            name: self._data_frame[col].dropna().to_numpy()
            for name, col in group_column_map.items()
        }

        return OneWayAnovaCalculator().calculate(
            groups=groups,
            significance_level=kwargs.get("significance_level", 0.05),
            run_post_hoc=kwargs.get("run_post_hoc", True),
        )

class AnalyseChiSquare(BaseDataAnalysis):
    """Chi-square independence test between two categorical columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_a="gender",
            column_b="purchased",
            significance_level=0.05,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_a: str = kwargs.get("column_a")
        column_b: str = kwargs.get("column_b")

        if column_a is None or column_b is None:
            raise ValueError("'column_a' and 'column_b' are required.")
        for col in (column_a, column_b):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return ChiSquareTestCalculator().calculate(
            series_a=self._data_frame[column_a].dropna(),
            series_b=self._data_frame[column_b].dropna(),
            significance_level=kwargs.get("significance_level", 0.05),
        )

class AnalyseCorrelationSignificance(BaseDataAnalysis):
    """Correlation with significance test and confidence interval.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_x="feature",
            column_y="target",
            method="pearson",        # "pearson" | "spearman"
            significance_level=0.05,
            confidence_level=0.95,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_x: str = kwargs.get("column_x")
        column_y: str = kwargs.get("column_y")

        if column_x is None or column_y is None:
            raise ValueError("'column_x' and 'column_y' are required.")
        for col in (column_x, column_y):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        paired = self._data_frame[[column_x, column_y]].dropna()

        return CorrelationSignificanceCalculator().calculate(
            x=paired[column_x].to_numpy(),
            y=paired[column_y].to_numpy(),
            method=kwargs.get("method", "pearson"),
            significance_level=kwargs.get("significance_level", 0.05),
            confidence_level=kwargs.get("confidence_level", 0.95),
        )

class AnalyseConfidenceIntervals(BaseDataAnalysis):
    """Compute confidence intervals for mean, proportion, or mean difference.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)

        # Mean CI
        result = analyzer.analyze(
            ci_type="mean",
            column="revenue",
            confidence_level=0.95,
        )

        # Proportion CI
        result = analyzer.analyze(
            ci_type="proportion",
            column="converted",   # binary 0/1 column
            confidence_level=0.95,
        )

        # Mean difference CI
        result = analyzer.analyze(
            ci_type="mean_difference",
            column_a="group_control",
            column_b="group_treatment",
            confidence_level=0.95,
        )
    """

    def analyze(self, **kwargs) -> dict:
        ci_type: str = kwargs.get("ci_type", "mean")
        confidence_level: float = kwargs.get("confidence_level", 0.95)
        calculator = ConfidenceIntervalCalculator()

        if ci_type == "mean":
            column: str = kwargs.get("column")
            if column is None:
                raise ValueError("'column' is required for ci_type='mean'.")
            if column not in self._data_frame.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame.")
            data = self._data_frame[column].dropna().to_numpy()
            return calculator.calculate("mean", confidence_level=confidence_level, data=data)

        if ci_type == "proportion":
            column = kwargs.get("column")
            if column is None:
                raise ValueError("'column' is required for ci_type='proportion'.")
            if column not in self._data_frame.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame.")
            series = self._data_frame[column].dropna()
            return calculator.calculate(
                "proportion",
                confidence_level=confidence_level,
                n_successes=int(series.sum()),
                n_total=len(series),
            )

        if ci_type == "mean_difference":
            col_a: str = kwargs.get("column_a")
            col_b: str = kwargs.get("column_b")
            if col_a is None or col_b is None:
                raise ValueError(
                    "'column_a' and 'column_b' are required for ci_type='mean_difference'."
                )
            for col in (col_a, col_b):
                if col not in self._data_frame.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame.")
            return calculator.calculate(
                "mean_difference",
                confidence_level=confidence_level,
                group_a=self._data_frame[col_a].dropna().to_numpy(),
                group_b=self._data_frame[col_b].dropna().to_numpy(),
            )

        raise ValueError(
            f"ci_type '{ci_type}' not recognized. "
            f"Available: 'mean', 'proportion', 'mean_difference'."
        )

class AnalyseEffectSize(BaseDataAnalysis):
    """Calculate effect size (Cohen's d or Eta-squared) from DataFrame columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)

        # Cohen's d
        result = analyzer.analyze(
            effect_type="cohens_d",
            column_a="control",
            column_b="treatment",
        )

        # Eta-squared
        result = analyzer.analyze(
            effect_type="eta_squared",
            groups={"control": "col_a", "t1": "col_b", "t2": "col_c"},
        )
    """

    def analyze(self, **kwargs) -> dict:
        effect_type: str = kwargs.get("effect_type", "cohens_d")

        if effect_type == "cohens_d":
            col_a: str = kwargs.get("column_a")
            col_b: str = kwargs.get("column_b")
            if col_a is None or col_b is None:
                raise ValueError("'column_a' and 'column_b' are required.")
            for col in (col_a, col_b):
                if col not in self._data_frame.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame.")
            return EffectSizeCalculator().calculate(
                "cohens_d",
                group_a=self._data_frame[col_a].dropna().to_numpy(),
                group_b=self._data_frame[col_b].dropna().to_numpy(),
            )

        if effect_type == "eta_squared":
            group_col_map: dict[str, str] = kwargs.get("groups")
            if group_col_map is None:
                raise ValueError("'groups' dict is required for eta_squared.")
            missing = [c for c in group_col_map.values() if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Columns not found: {missing}")
            groups = {
                name: self._data_frame[col].dropna().to_numpy()
                for name, col in group_col_map.items()
            }
            return EffectSizeCalculator().calculate("eta_squared", groups=groups)

        raise ValueError(
            f"effect_type '{effect_type}' not recognized. "
            f"Available: 'cohens_d', 'eta_squared'."
        )

class AnalysePowerAnalysis(BaseDataAnalysis):
    """Statistical power analysis from DataFrame context.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)

        result = analyzer.analyze(
            analysis_type="minimum_n",
            effect_size=0.5,
            alpha=0.05,
            target_power=0.80,
        )

        result = analyzer.analyze(
            analysis_type="observed_power",
            column_a="control",
            column_b="treatment",
            alpha=0.05,
        )
    """

    def analyze(self, **kwargs) -> dict:
        analysis_type: str = kwargs.get("analysis_type", "minimum_n")
        alpha: float = kwargs.get("alpha", 0.05)

        if analysis_type == "minimum_n":
            return PowerAnalysisCalculator().calculate(
                "minimum_n",
                effect_size=kwargs["effect_size"],
                alpha=alpha,
                target_power=kwargs.get("target_power", 0.80),
            )

        if analysis_type == "observed_power":
            col_a: str = kwargs.get("column_a")
            col_b: str = kwargs.get("column_b")
            if col_a is None or col_b is None:
                raise ValueError(
                    "'column_a' and 'column_b' are required for observed_power."
                )
            for col in (col_a, col_b):
                if col not in self._data_frame.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame.")

            from statistics.inferential.effect_size import CohensDCalculator
            group_a = self._data_frame[col_a].dropna().to_numpy()
            group_b = self._data_frame[col_b].dropna().to_numpy()
            d = CohensDCalculator().calculate(group_a, group_b)
            n_per_group = min(len(group_a), len(group_b))

            return PowerAnalysisCalculator().calculate(
                "observed_power",
                effect_size=abs(d),
                n_per_group=n_per_group,
                alpha=alpha,
            )

        raise ValueError(
            f"analysis_type '{analysis_type}' not recognized. "
            f"Available: 'minimum_n', 'observed_power'."
        )

class AnalyseBootstrap(BaseDataAnalysis):
    """Non-parametric bootstrap CI for any statistic on a column.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="revenue",
            statistic=np.median,   # any callable: np.mean, np.std, etc.
            n_iterations=5000,
            confidence_level=0.95,
            random_seed=42,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        statistic = kwargs.get("statistic", np.mean)

        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        data = self._data_frame[column].dropna().to_numpy()

        return BootstrapEstimator().estimate(
            data=data,
            statistic=statistic,
            n_iterations=kwargs.get("n_iterations", 5_000),
            confidence_level=kwargs.get("confidence_level", 0.95),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalyseCorrelationMatrix(BaseDataAnalysis):
    """Full correlation matrix with ranked pairs and multicollinearity flags.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            columns=["col_a", "col_b", "col_c"],  # optional
            method="pearson",                       # "pearson"|"spearman"|"kendall"
            top_n=10,
            threshold=0.0,
            high_correlation_flag=0.85,
        )
    """

    def analyze(self, **kwargs) -> dict:
        columns: list[str] | None = kwargs.get("columns")
        numeric_df = self._data_frame.select_dtypes(include=[np.number])

        if columns is not None:
            missing = [c for c in columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {missing}")
            numeric_df = numeric_df[columns]

        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation analysis.")

        return CorrelationMatrixCalculator().calculate(
            data_frame=numeric_df,
            method=kwargs.get("method", "pearson"),
            top_n=kwargs.get("top_n"),
            threshold=kwargs.get("threshold", 0.0),
            high_correlation_flag=kwargs.get("high_correlation_flag", 0.85),
        )

class AnalyseMulticollinearity(BaseDataAnalysis):
    """VIF-based multicollinearity detection for linear model preparation.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            columns=["age", "income", "credit_score"],  # optional
            high_vif_threshold=10.0,
        )
    """

    def analyze(self, **kwargs) -> dict:
        columns: list[str] | None = kwargs.get("columns")
        numeric_df = self._data_frame.select_dtypes(include=[np.number])

        if columns is not None:
            missing = [c for c in columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Columns not found in DataFrame: {missing}")
            numeric_df = numeric_df[columns]

        if numeric_df.empty:
            raise ValueError("No numeric columns found for VIF analysis.")

        return MulticollinearityCalculator().calculate(
            data_frame=numeric_df,
            high_vif_threshold=kwargs.get("high_vif_threshold", 10.0),
        )

class AnalyseMutualInformation(BaseDataAnalysis):
    """MI-based feature relevance scoring against a target variable.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            target_column="churn",
            feature_columns=["age", "income", "region"],  # optional
            target_type="auto",   # "auto"|"categorical"|"continuous"
            top_n=10,
            random_seed=42,
        )
    """

    def analyze(self, **kwargs) -> dict:
        target_column: str = kwargs.get("target_column")
        feature_columns: list[str] | None = kwargs.get("feature_columns")

        if target_column is None:
            raise ValueError("'target_column' is required.")
        if target_column not in self._data_frame.columns:
            raise KeyError(f"Column '{target_column}' not found in DataFrame.")

        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Feature columns not found: {missing}")
            features_df = self._data_frame[feature_columns]
        else:
            features_df = self._data_frame.drop(columns=[target_column])

        return MutualInformationCalculator().calculate(
            features=features_df,
            target=self._data_frame[target_column],
            target_type=kwargs.get("target_type", "auto"),
            top_n=kwargs.get("top_n"),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalysePartialCorrelation(BaseDataAnalysis):
    """Partial correlation between two columns controlling for confounders.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_x="income",
            column_y="savings",
            control_columns=["age", "education_years"],
            significance_level=0.05,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_x: str = kwargs.get("column_x")
        column_y: str = kwargs.get("column_y")
        control_columns: list[str] = kwargs.get("control_columns", [])

        if column_x is None or column_y is None:
            raise ValueError("'column_x' and 'column_y' are required.")
        if not control_columns:
            raise ValueError(
                "'control_columns' must contain at least one variable to control for."
            )

        return PartialCorrelationCalculator().calculate(
            data_frame=self._data_frame,
            column_x=column_x,
            column_y=column_y,
            control_columns=control_columns,
            significance_level=kwargs.get("significance_level", 0.05),
        )

class AnalyseCrossCorrelation(BaseDataAnalysis):
    """Cross-correlation between two time series across a range of lags.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_x="advertising_spend",
            column_y="sales",
            max_lag=12,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_x: str = kwargs.get("column_x")
        column_y: str = kwargs.get("column_y")

        if column_x is None or column_y is None:
            raise ValueError("'column_x' and 'column_y' are required.")

        return CrossCorrelationCalculator().calculate(
            data_frame=self._data_frame,
            column_x=column_x,
            column_y=column_y,
            max_lag=kwargs.get("max_lag", 10),
        )

class AnalyseGrangerCausality(BaseDataAnalysis):
    """Granger causality test: does x improve forecasts of y?

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_y="sales",
            column_x="advertising",
            max_lag=4,
            significance_level=0.05,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_y: str = kwargs.get("column_y")
        column_x: str = kwargs.get("column_x")

        if column_y is None or column_x is None:
            raise ValueError("'column_y' and 'column_x' are required.")

        return GrangerCausalityCalculator().calculate(
            data_frame=self._data_frame,
            column_y=column_y,
            column_x=column_x,
            max_lag=kwargs.get("max_lag", 4),
            significance_level=kwargs.get("significance_level", 0.05),
        )

class AnalyseContingency(BaseDataAnalysis):
    """Full 2×2 contingency analysis: chi-square, OR, RR, Cramér's V.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_exposure="vaccinated",
            column_outcome="infected",
            significance_level=0.05,
            confidence_level=0.95,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_exposure: str = kwargs.get("column_exposure")
        column_outcome: str = kwargs.get("column_outcome")

        if column_exposure is None or column_outcome is None:
            raise ValueError("'column_exposure' and 'column_outcome' are required.")

        return ContingencyAnalysisCalculator().calculate(
            data_frame=self._data_frame,
            column_exposure=column_exposure,
            column_outcome=column_outcome,
            significance_level=kwargs.get("significance_level", 0.05),
            confidence_level=kwargs.get("confidence_level", 0.95),
        )

class AnalyseInteractionEffects(BaseDataAnalysis):
    """Detect feature pairs whose interaction improves target R².

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            target_column="price",
            feature_columns=["sqft", "rooms", "age"],  # optional
            min_gain_threshold=0.01,
            top_n=10,
        )
    """

    def analyze(self, **kwargs) -> dict:
        target_column: str = kwargs.get("target_column")
        if target_column is None:
            raise ValueError("'target_column' is required.")

        return InteractionEffectsCalculator().calculate(
            data_frame=self._data_frame,
            target_column=target_column,
            feature_columns=kwargs.get("feature_columns"),
            min_gain_threshold=kwargs.get("min_gain_threshold", 0.01),
            top_n=kwargs.get("top_n"),
        )

# ── DOMAIN 5 IMPORTS ──────────────────────────────────────────────────────────
from statistics.ml_support.feature_variance import FeatureVarianceCalculator
from statistics.ml_support.feature_selection import FeatureSelectionCalculator
from statistics.ml_support.feature_importance import FeatureImportanceCalculator
from statistics.ml_support.dimensionality_reduction import DimensionalityReductionCalculator
from statistics.ml_support.class_imbalance import ClassImbalanceCalculator
from statistics.ml_support.model_residuals import ModelResidualsCalculator
from statistics.ml_support.learning_curve import LearningCurveCalculator
from statistics.ml_support.cross_validation import CrossValidationCalculator

class AnalyseFeatureVariance(BaseDataAnalysis):
    """Near-zero variance detection across all numeric columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            variance_threshold=1e-4,
            unique_ratio_threshold=0.01,
            frequency_ratio_threshold=0.95,
        )
    """

    def analyze(self, **kwargs) -> dict:
        return FeatureVarianceCalculator().calculate(
            data_frame=self._data_frame,
            variance_threshold=kwargs.get("variance_threshold", 1e-4),
            unique_ratio_threshold=kwargs.get("unique_ratio_threshold", 0.01),
            frequency_ratio_threshold=kwargs.get("frequency_ratio_threshold", 0.95),
        )

class AnalyseFeatureSelection(BaseDataAnalysis):
    """Univariate feature scoring: chi2, ANOVA F, mutual information.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            target_column="churn",
            feature_columns=["age", "income"],
            methods=["chi2", "anova_f", "mutual_information"],
            target_type="classification",
            top_n=10,
        )
    """

    def analyze(self, **kwargs) -> dict:
        target_column: str = kwargs.get("target_column")
        if target_column is None:
            raise ValueError("'target_column' is required.")
        if target_column not in self._data_frame.columns:
            raise KeyError(f"Column '{target_column}' not found.")

        return FeatureSelectionCalculator().calculate(
            data_frame=self._data_frame,
            target_column=target_column,
            feature_columns=kwargs.get("feature_columns"),
            methods=kwargs.get("methods"),
            target_type=kwargs.get("target_type", "classification"),
            top_n=kwargs.get("top_n"),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalyseFeatureImportance(BaseDataAnalysis):
    """Random Forest Gini and permutation importance.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            target_column="churn",
            feature_columns=["age", "income"],
            methods=["gini", "permutation"],
            target_type="classification",
            n_estimators=100,
            top_n=15,
        )
    """

    def analyze(self, **kwargs) -> dict:
        target_column: str = kwargs.get("target_column")
        if target_column is None:
            raise ValueError("'target_column' is required.")
        if target_column not in self._data_frame.columns:
            raise KeyError(f"Column '{target_column}' not found.")

        return FeatureImportanceCalculator().calculate(
            data_frame=self._data_frame,
            target_column=target_column,
            feature_columns=kwargs.get("feature_columns"),
            methods=kwargs.get("methods"),
            target_type=kwargs.get("target_type", "classification"),
            n_estimators=kwargs.get("n_estimators", 100),
            n_repeats=kwargs.get("n_repeats", 10),
            top_n=kwargs.get("top_n"),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalyseDimensionalityReduction(BaseDataAnalysis):
    """PCA with variance explained, loadings, and optimal component selection.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            columns=["age", "income", "score"],  # optional
            n_components=None,
            target_variance_explained=0.95,
        )
    """

    def analyze(self, **kwargs) -> dict:
        columns: list[str] | None = kwargs.get("columns")
        numeric_df = self._data_frame.select_dtypes(include=[np.number])

        if columns is not None:
            missing = [c for c in columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Columns not found: {missing}")
            numeric_df = numeric_df[columns]

        if numeric_df.empty:
            raise ValueError("No numeric columns found for PCA.")

        return DimensionalityReductionCalculator().calculate(
            data_frame=numeric_df,
            n_components=kwargs.get("n_components"),
            target_variance_explained=kwargs.get("target_variance_explained", 0.95),
        )

class AnalyseClassImbalance(BaseDataAnalysis):
    """Class distribution analysis with resampling strategy recommendation.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            target_column="churn",
            minority_threshold=0.3,
        )
    """

    def analyze(self, **kwargs) -> dict:
        target_column: str = kwargs.get("target_column")
        if target_column is None:
            raise ValueError("'target_column' is required.")
        if target_column not in self._data_frame.columns:
            raise KeyError(f"Column '{target_column}' not found.")

        return ClassImbalanceCalculator().calculate(
            series=self._data_frame[target_column],
            minority_threshold=kwargs.get("minority_threshold", 0.3),
        )

class AnalyseModelResiduals(BaseDataAnalysis):
    """Residual diagnostics: normality, homoscedasticity, autocorrelation.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            actual_column="actual_sales",
            predicted_column="predicted_sales",
            significance_level=0.05,
        )
    """

    def analyze(self, **kwargs) -> dict:
        actual_column: str = kwargs.get("actual_column")
        predicted_column: str = kwargs.get("predicted_column")

        if actual_column is None or predicted_column is None:
            raise ValueError("'actual_column' and 'predicted_column' are required.")
        for col in (actual_column, predicted_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found.")

        paired = self._data_frame[[actual_column, predicted_column]].dropna()
        return ModelResidualsCalculator().calculate(
            y_true=paired[actual_column].to_numpy(dtype=float),
            y_pred=paired[predicted_column].to_numpy(dtype=float),
            significance_level=kwargs.get("significance_level", 0.05),
        )

class AnalyseLearningCurve(BaseDataAnalysis):
    """Learning curve for bias-variance diagnosis across training sizes.

    Workflow:
        from sklearn.linear_model import LogisticRegression

        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            target_column="churn",
            estimator=LogisticRegression(),
            feature_columns=["age", "income"],
            strategy="stratified_kfold",
            n_checkpoints=10,
            cv=5,
            scoring="accuracy",
            gap_threshold=0.1,
            min_acceptable_score=0.6,
        )
    """

    def analyze(self, **kwargs) -> dict:
        target_column: str = kwargs.get("target_column")
        estimator = kwargs.get("estimator")

        if target_column is None:
            raise ValueError("'target_column' is required.")
        if estimator is None:
            raise ValueError("'estimator' is required (sklearn-compatible model).")
        if target_column not in self._data_frame.columns:
            raise KeyError(f"Column '{target_column}' not found.")

        return LearningCurveCalculator().calculate(
            data_frame=self._data_frame,
            target_column=target_column,
            estimator=estimator,
            feature_columns=kwargs.get("feature_columns"),
            n_checkpoints=kwargs.get("n_checkpoints", 10),
            cv=kwargs.get("cv", 5),
            scoring=kwargs.get("scoring", "accuracy"),
            gap_threshold=kwargs.get("gap_threshold", 0.1),
            min_acceptable_score=kwargs.get("min_acceptable_score", 0.6),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalyseCrossValidation(BaseDataAnalysis):
    """K-Fold / Stratified / Repeated cross-validation with CI.

    Workflow:
        from sklearn.ensemble import RandomForestClassifier

        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            target_column="churn",
            estimator=RandomForestClassifier(n_estimators=100),
            feature_columns=["age", "income"],
            strategy="stratified_kfold",
            n_folds=5,
            n_repeats=3,
            scoring="f1_weighted",
            confidence_level=0.95,
        )
    """

    def analyze(self, **kwargs) -> dict:
        target_column: str = kwargs.get("target_column")
        estimator = kwargs.get("estimator")

        if target_column is None:
            raise ValueError("'target_column' is required.")
        if estimator is None:
            raise ValueError("'estimator' is required (sklearn-compatible model).")
        if target_column not in self._data_frame.columns:
            raise KeyError(f"Column '{target_column}' not found.")

        return CrossValidationCalculator().calculate(
            data_frame=self._data_frame,
            target_column=target_column,
            estimator=estimator,
            feature_columns=kwargs.get("feature_columns"),
            strategy=kwargs.get("strategy", "stratified_kfold"),
            n_folds=kwargs.get("n_folds", 5),
            n_repeats=kwargs.get("n_repeats", 3),
            scoring=kwargs.get("scoring", "accuracy"),
            confidence_level=kwargs.get("confidence_level", 0.95),
            random_seed=kwargs.get("random_seed", 42),
            n_jobs=kwargs.get("n_jobs", -1),
        )

# ── DOMAIN 4 IMPORTS ──────────────────────────────────────────────────────────
from statistics.time_series.volatility import VolatilityCalculator
from statistics.time_series.momentum import MomentumCalculator
from statistics.time_series.moving_averages import MovingAveragesCalculator
from statistics.time_series.stationarity import StationarityCalculator
from statistics.time_series.lag_features import LagFeaturesCalculator
from statistics.time_series.change_points import ChangePointDetector
from statistics.time_series.forecast_accuracy import ForecastAccuracyCalculator
from statistics.time_series.cyclical_patterns import CyclicalPatternsCalculator
from statistics.time_series.rolling_statistics import RollingStatisticsCalculator

class AnalyseVolatility(BaseDataAnalysis):
    """Rolling std, EWMA volatility, CV and regime detection."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        series = self._data_frame[column].dropna().to_numpy(dtype=float)
        return VolatilityCalculator().calculate(
            series=series,
            window=kwargs.get("window", 20),
            decay_factor=kwargs.get("decay_factor", 0.94),
        )

class AnalyseMomentum(BaseDataAnalysis):
    """Rate of change, acceleration, and momentum signal classification."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        series = self._data_frame[column].dropna().to_numpy(dtype=float)
        return MomentumCalculator().calculate(
            series=series,
            period=kwargs.get("period", 14),
        )

class AnalyseMovingAverages(BaseDataAnalysis):
    """SMA, EMA, WMA computation with optional crossover detection."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        series = self._data_frame[column].dropna().to_numpy(dtype=float)
        return MovingAveragesCalculator().calculate(
            series=series,
            periods=kwargs.get("periods", [20, 50, 200]),
            ma_types=kwargs.get("ma_types"),
            detect_crossovers=kwargs.get("detect_crossovers", False),
            fast_period=kwargs.get("fast_period"),
            slow_period=kwargs.get("slow_period"),
            crossover_ma_type=kwargs.get("crossover_ma_type", "ema"),
        )

class AnalyseStationarity(BaseDataAnalysis):
    """ADF + KPSS combined stationarity test with recommendation."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        series = self._data_frame[column].dropna().to_numpy(dtype=float)
        return StationarityCalculator().calculate(
            series=series,
            max_lags=kwargs.get("max_lags", 4),
            significance_level=kwargs.get("significance_level", 0.05),
        )

class AnalyseLagFeatures(BaseDataAnalysis):
    """ACF, PACF analysis and lag feature generation."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        return LagFeaturesCalculator().calculate(
            series=self._data_frame[column],
            max_lag=kwargs.get("max_lag", 20),
            lags_to_generate=kwargs.get("lags_to_generate"),
            significance_level=kwargs.get("significance_level", 0.05),
        )

class AnalyseChangePoints(BaseDataAnalysis):
    """CUSUM mean-shift and variance change point detection."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        series = self._data_frame[column].dropna().to_numpy(dtype=float)
        return ChangePointDetector().calculate(
            series=series,
            k_multiplier=kwargs.get("k_multiplier", 0.5),
            h_multiplier=kwargs.get("h_multiplier", 4.0),
            variance_ratio_threshold=kwargs.get("variance_ratio_threshold", 2.0),
        )

class AnalyseForecastAccuracy(BaseDataAnalysis):
    """MAE, RMSE, MAPE, MASE forecast accuracy metrics."""

    def analyze(self, **kwargs) -> dict:
        actual_column: str = kwargs.get("actual_column")
        predicted_column: str = kwargs.get("predicted_column")
        if actual_column is None or predicted_column is None:
            raise ValueError("'actual_column' and 'predicted_column' are required.")
        for col in (actual_column, predicted_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        paired = self._data_frame[[actual_column, predicted_column]].dropna()
        return ForecastAccuracyCalculator().calculate(
            y_true=paired[actual_column].to_numpy(dtype=float),
            y_pred=paired[predicted_column].to_numpy(dtype=float),
            metrics=kwargs.get("metrics"),
        )

class AnalyseCyclicalPatterns(BaseDataAnalysis):
    """FFT-based dominant cycle detection."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        series = self._data_frame[column].dropna().to_numpy(dtype=float)
        return CyclicalPatternsCalculator().calculate(
            series=series,
            top_n=kwargs.get("top_n", 5),
            remove_trend=kwargs.get("remove_trend", True),
            apply_window=kwargs.get("apply_window", True),
        )

class AnalyseRollingStatistics(BaseDataAnalysis):
    """Configurable rolling statistics: mean, std, min, max, median, skewness."""

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        series = self._data_frame[column].dropna().to_numpy(dtype=float)
        return RollingStatisticsCalculator().calculate(
            series=series,
            window=kwargs.get("window", 20),
            statistics=kwargs.get("statistics"),
        )

class AnalyseTextBasicStats(BaseDataAnalysis):
    """Per-document and corpus-level text statistics.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="review_text",
            sample_n=None,   # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        return TextBasicStatsCalculator().calculate(
            series=self._data_frame[column],
            sample_n=kwargs.get("sample_n"),
        )

class AnalyseWordFrequency(BaseDataAnalysis):
    """TF and TF-IDF term frequency ranking across a text corpus.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="article_body",
            top_n=30,
            remove_stopwords=True,
            custom_stopwords=None,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        return WordFrequencyCalculator().calculate(
            series=self._data_frame[column],
            top_n=kwargs.get("top_n", 30),
            remove_stopwords=kwargs.get("remove_stopwords", True),
            custom_stopwords=kwargs.get("custom_stopwords"),
        )

class AnalyseSentiment(BaseDataAnalysis):
    """Lexicon-based polarity and subjectivity analysis.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="user_review",
            sample_n=None,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        return SentimentAnalysisCalculator().calculate(
            series=self._data_frame[column],
            sample_n=kwargs.get("sample_n"),
        )

class AnalyseTopicDetection(BaseDataAnalysis):
    """NMF-based latent topic discovery in a text corpus.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="article_body",
            n_topics=5,
            top_terms_per_topic=10,
            min_df=2,
            max_df_ratio=0.9,
            random_seed=42,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        return TopicDetectionCalculator().calculate(
            series=self._data_frame[column],
            n_topics=kwargs.get("n_topics", 5),
            top_terms_per_topic=kwargs.get("top_terms_per_topic", 10),
            min_df=kwargs.get("min_df", 2),
            max_df_ratio=kwargs.get("max_df_ratio", 0.9),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalyseLanguageDetection(BaseDataAnalysis):
    """Character trigram-based language detection per document.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="user_comment",
            top_n_candidates=3,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        return LanguageDetectionCalculator().calculate(
            series=self._data_frame[column],
            top_n_candidates=kwargs.get("top_n_candidates", 3),
        )

class AnalyseTextSimilarity(BaseDataAnalysis):
    """Pairwise TF-IDF cosine similarity between two text columns.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column_a="original_text",
            column_b="translated_text",
        )
    """

    def analyze(self, **kwargs) -> dict:
        column_a: str = kwargs.get("column_a")
        column_b: str = kwargs.get("column_b")

        if column_a is None or column_b is None:
            raise ValueError("'column_a' and 'column_b' are required.")
        for col in (column_a, column_b):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return TextSimilarityCalculator().calculate(
            data_frame=self._data_frame,
            column_a=column_a,
            column_b=column_b,
        )

class AnalyseNamedEntityDensity(BaseDataAnalysis):
    """Rule-based named entity detection and density analysis.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            column="article_body",
            entity_types=["PERSON", "ORGANIZATION", "MONEY"],  # optional
            sample_n=None,
        )
    """

    def analyze(self, **kwargs) -> dict:
        column: str = kwargs.get("column")
        if column is None:
            raise ValueError("'column' is required.")
        if column not in self._data_frame.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        return NamedEntityDensityCalculator().calculate(
            series=self._data_frame[column],
            entity_types=kwargs.get("entity_types"),
            sample_n=kwargs.get("sample_n"),
        )

# ── DOMAIN 7 — SEGMENTATION ───────────────────────────────────────────────────

class AnalyseKMeansClusters(BaseDataAnalysis):
    """K-Means clustering with auto K selection via silhouette score.

    Workflow:
        analyzer = AnalyzerFactory.create("kmeans_clusters", df)
        result = analyzer.analyze(
            feature_columns=["recency", "frequency", "monetary"],
            n_clusters=None,     # auto-selects optimal K
            k_range=(2, 8),
            random_seed=42,
        )
    """

    def analyze(self, **kwargs) -> dict:
        feature_columns: list[str] | None = kwargs.get("feature_columns")
        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Feature columns not found: {missing}")
        return KMeansClusterCalculator().calculate(
            data_frame=self._data_frame,
            feature_columns=feature_columns,
            n_clusters=kwargs.get("n_clusters"),
            k_range=kwargs.get("k_range", (2, 8)),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalyseRFMSegmentation(BaseDataAnalysis):
    """RFM segmentation from transactional data.

    Workflow:
        analyzer = AnalyzerFactory.create("rfm_segmentation", df)
        result = analyzer.analyze(
            customer_column="customer_id",
            date_column="purchase_date",
            amount_column="order_value",
            reference_date=None,   # optional pd.Timestamp
        )
    """

    def analyze(self, **kwargs) -> dict:
        required = ["customer_column", "date_column", "amount_column"]
        missing_params = [p for p in required if kwargs.get(p) is None]
        if missing_params:
            raise ValueError(f"Required parameters: {missing_params}")
        for col in (
            kwargs["customer_column"],
            kwargs["date_column"],
            kwargs["amount_column"],
        ):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        return RFMSegmentationCalculator().calculate(
            data_frame=self._data_frame,
            customer_column=kwargs["customer_column"],
            date_column=kwargs["date_column"],
            amount_column=kwargs["amount_column"],
            reference_date=kwargs.get("reference_date"),
        )

class AnalyseCohortAnalysis(BaseDataAnalysis):
    """Cohort retention matrix by acquisition period.

    Workflow:
        analyzer = AnalyzerFactory.create("cohort_analysis", df)
        result = analyzer.analyze(
            user_column="user_id",
            date_column="activity_date",
            period="M",    # "M" | "W" | "Q"
        )
    """

    def analyze(self, **kwargs) -> dict:
        user_column: str = kwargs.get("user_column")
        date_column: str = kwargs.get("date_column")
        if user_column is None or date_column is None:
            raise ValueError("'user_column' and 'date_column' are required.")
        for col in (user_column, date_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        return CohortAnalysisCalculator().calculate(
            data_frame=self._data_frame,
            user_column=user_column,
            date_column=date_column,
            period=kwargs.get("period", "M"),
        )

class AnalysePopulationSplits(BaseDataAnalysis):
    """Statistical feature comparison between two population groups.

    Workflow:
        analyzer = AnalyzerFactory.create("population_splits", df)
        result = analyzer.analyze(
            split_column="is_churned",
            group_a_value=0,
            group_b_value=1,
            feature_columns=None,       # optional
            significance_level=0.05,
        )
    """

    def analyze(self, **kwargs) -> dict:
        split_column: str = kwargs.get("split_column")
        group_a_value = kwargs.get("group_a_value")
        group_b_value = kwargs.get("group_b_value")
        if split_column is None:
            raise ValueError("'split_column' is required.")
        if group_a_value is None or group_b_value is None:
            raise ValueError("'group_a_value' and 'group_b_value' are required.")
        if split_column not in self._data_frame.columns:
            raise KeyError(f"Column '{split_column}' not found in DataFrame.")
        return PopulationSplitsCalculator().calculate(
            data_frame=self._data_frame,
            split_column=split_column,
            group_a_value=group_a_value,
            group_b_value=group_b_value,
            feature_columns=kwargs.get("feature_columns"),
            significance_level=kwargs.get("significance_level", 0.05),
        )

class AnalyseDBSCANClusters(BaseDataAnalysis):
    """DBSCAN density-based clustering with auto epsilon estimation.

    Workflow:
        analyzer = AnalyzerFactory.create("dbscan_clusters", df)
        result = analyzer.analyze(
            feature_columns=["lat", "lon"],
            epsilon=None,       # auto-estimated via k-distance elbow
            min_samples=5,
        )
    """

    def analyze(self, **kwargs) -> dict:
        feature_columns: list[str] | None = kwargs.get("feature_columns")
        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Feature columns not found: {missing}")
        return DBSCANClusterCalculator().calculate(
            data_frame=self._data_frame,
            feature_columns=feature_columns,
            epsilon=kwargs.get("epsilon"),
            min_samples=kwargs.get("min_samples", 5),
            random_seed=kwargs.get("random_seed", 42),
        )

class AnalyseHierarchicalClusters(BaseDataAnalysis):
    """Hierarchical agglomerative clustering with cophenetic correlation.

    Workflow:
        analyzer = AnalyzerFactory.create("hierarchical_clusters", df)
        result = analyzer.analyze(
            feature_columns=["age", "income", "score"],
            n_clusters=None,
            k_range=(2, 8),
            linkage_method="ward",    # "ward"|"complete"|"average"|"single"
            extract_dendrogram=False,
        )
    """

    def analyze(self, **kwargs) -> dict:
        feature_columns: list[str] | None = kwargs.get("feature_columns")
        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in self._data_frame.columns]
            if missing:
                raise KeyError(f"Feature columns not found: {missing}")
        return HierarchicalClusterCalculator().calculate(
            data_frame=self._data_frame,
            feature_columns=feature_columns,
            n_clusters=kwargs.get("n_clusters"),
            k_range=kwargs.get("k_range", (2, 8)),
            linkage_method=kwargs.get("linkage_method", "ward"),
            extract_dendrogram=kwargs.get("extract_dendrogram", False),
        )

class AnalyseGrowthRates(BaseDataAnalysis):
    """MoM/YoY period-over-period growth, CAGR, and rolling growth.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            value_column="revenue",
            date_column="month",        # optional
            period_window=1,
            periods_per_year=12,
        )
    """

    def analyze(self, **kwargs) -> dict:
        value_column: str = kwargs.get("value_column")
        if value_column is None:
            raise ValueError("'value_column' is required.")
        if value_column not in self._data_frame.columns:
            raise KeyError(f"Column '{value_column}' not found in DataFrame.")

        return GrowthRatesCalculator().calculate(
            data_frame=self._data_frame,
            value_column=value_column,
            date_column=kwargs.get("date_column"),
            period_window=kwargs.get("period_window", 1),
            n_years=kwargs.get("n_years"),
            periods_per_year=kwargs.get("periods_per_year", 12),
        )

class AnalyseRiskMetrics(BaseDataAnalysis):
    """VaR, CVaR, Sharpe, Sortino, Max Drawdown, and Calmar ratio.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            value_column="portfolio_value",
            returns_column=None,          # optional pre-computed returns
            confidence_level=0.95,
            risk_free_rate=0.02,
            periods_per_year=252,
        )
    """

    def analyze(self, **kwargs) -> dict:
        value_column: str = kwargs.get("value_column")
        if value_column is None:
            raise ValueError("'value_column' is required.")
        if value_column not in self._data_frame.columns:
            raise KeyError(f"Column '{value_column}' not found in DataFrame.")

        return RiskMetricsCalculator().calculate(
            data_frame=self._data_frame,
            value_column=value_column,
            returns_column=kwargs.get("returns_column"),
            returns_method=kwargs.get("returns_method", "simple"),
            confidence_level=kwargs.get("confidence_level", 0.95),
            risk_free_rate=kwargs.get("risk_free_rate", 0.02),
            periods_per_year=kwargs.get("periods_per_year", 252),
        )

class AnalyseFinancialRatios(BaseDataAnalysis):
    """Profitability, liquidity, leverage, and efficiency ratios.

    Required columns (subset used per ratio):
        revenue, cogs, net_income, shareholders_equity,
        total_assets, current_assets, current_liabilities,
        inventory, total_debt

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            ratios=["gross_margin", "roe", "current_ratio"],  # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        return FinancialRatiosCalculator().calculate(
            data_frame=self._data_frame,
            ratios=kwargs.get("ratios"),
        )

class AnalyseConversionFunnel(BaseDataAnalysis):
    """Funnel analysis with stage conversion rates and bottleneck detection.

    Workflow — pre-aggregated:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            stage_counts={"Visit": 10000, "Signup": 3000, "Purchase": 500},
            bottleneck_threshold=0.5,
        )

    Workflow — event log:
        result = analyzer.analyze(
            user_column="user_id",
            event_column="event_name",
            stage_order=["Visit", "Signup", "Purchase"],
            bottleneck_threshold=0.5,
        )
    """

    def analyze(self, **kwargs) -> dict:
        stage_counts: dict | None = kwargs.get("stage_counts")
        user_column: str | None = kwargs.get("user_column")
        event_column: str | None = kwargs.get("event_column")
        stage_order: list[str] | None = kwargs.get("stage_order")

        if stage_counts is None and (
            user_column is None or event_column is None or stage_order is None
        ):
            raise ValueError(
                "Provide 'stage_counts' OR all of: 'user_column', "
                "'event_column', 'stage_order'."
            )

        return ConversionFunnelCalculator().calculate(
            stage_counts=stage_counts,
            data_frame=self._data_frame if stage_counts is None else None,
            user_column=user_column,
            event_column=event_column,
            stage_order=stage_order,
            bottleneck_threshold=kwargs.get("bottleneck_threshold", 0.5),
        )

class AnalyseChurnRate(BaseDataAnalysis):
    """Period-level churn rate from aggregated data or event logs.

    Workflow — aggregated:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            period_column="month",
            customers_start_column="customers_start",
            churned_column="churned",
            new_customers_column="new_customers",
            mode="aggregated",
        )

    Workflow — event log:
        result = analyzer.analyze(
            user_column="user_id",
            period_column="month",
            mode="events",
        )
    """

    def analyze(self, **kwargs) -> dict:
        period_column: str = kwargs.get("period_column")
        if period_column is None:
            raise ValueError("'period_column' is required.")
        if period_column not in self._data_frame.columns:
            raise KeyError(f"Column '{period_column}' not found in DataFrame.")

        return ChurnRateCalculator().calculate(
            data_frame=self._data_frame,
            period_column=period_column,
            mode=kwargs.get("mode", "aggregated"),
            customers_start_column=kwargs.get("customers_start_column"),
            churned_column=kwargs.get("churned_column"),
            new_customers_column=kwargs.get("new_customers_column"),
            user_column=kwargs.get("user_column"),
        )

class AnalyseCustomerLifetimeValue(BaseDataAnalysis):
    """Discounted and simple CLV per customer from transactional data.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            customer_column="customer_id",
            order_value_column="order_value",
            date_column="purchase_date",
            discount_rate=0.1,
            margin_rate=0.3,
            periods_per_year=12,
        )
    """

    def analyze(self, **kwargs) -> dict:
        required = ["customer_column", "order_value_column", "date_column"]
        missing_params = [p for p in required if kwargs.get(p) is None]
        if missing_params:
            raise ValueError(f"Required parameters: {missing_params}")

        for col_key in required:
            col = kwargs[col_key]
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return CustomerLifetimeValueCalculator().calculate(
            data_frame=self._data_frame,
            customer_column=kwargs["customer_column"],
            order_value_column=kwargs["order_value_column"],
            date_column=kwargs["date_column"],
            discount_rate=kwargs.get("discount_rate", 0.1),
            margin_rate=kwargs.get("margin_rate", 0.3),
            periods_per_year=kwargs.get("periods_per_year", 12),
        )

class AnalyseParetoAnalysis(BaseDataAnalysis):
    """Pareto (80/20) analysis with Gini concentration coefficient.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            entity_column="product_sku",
            value_column="revenue",
            target_share=0.8,
        )
    """

    def analyze(self, **kwargs) -> dict:
        entity_column: str = kwargs.get("entity_column")
        value_column: str = kwargs.get("value_column")

        if entity_column is None or value_column is None:
            raise ValueError("'entity_column' and 'value_column' are required.")
        for col in (entity_column, value_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return ParetoAnalysisCalculator().calculate(
            data_frame=self._data_frame,
            entity_column=entity_column,
            value_column=value_column,
            target_share=kwargs.get("target_share", 0.8),
        )

class AnalyseRunRate(BaseDataAnalysis):
    """Run rate projection: simple, trailing average, and weighted recent.

    Workflow — partial period (YTD):
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            observed_value=750_000,
            elapsed_fraction=0.625,
            methods=["simple"],
        )

    Workflow — historical periods:
        result = analyzer.analyze(
            value_column="monthly_revenue",
            full_periods=12,
            n_periods_trailing=3,
            decay=0.9,
            methods=["trailing_average", "weighted_recent"],
        )
    """

    def analyze(self, **kwargs) -> dict:
        value_column: str | None = kwargs.get("value_column")
        observed_value: float | None = kwargs.get("observed_value")

        if value_column is not None and value_column not in self._data_frame.columns:
            raise KeyError(f"Column '{value_column}' not found in DataFrame.")

        return RunRateCalculator().calculate(
            observed_value=observed_value,
            elapsed_fraction=kwargs.get("elapsed_fraction"),
            data_frame=self._data_frame if value_column is not None else None,
            value_column=value_column,
            full_periods=kwargs.get("full_periods", 12),
            n_periods_trailing=kwargs.get("n_periods_trailing", 3),
            decay=kwargs.get("decay", 0.9),
            methods=kwargs.get("methods"),
        )

class AnalyseGeoDistribution(BaseDataAnalysis):
    """Geographic frequency distribution with HHI concentration metric.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            geo_column="country",
            top_n=20,
            secondary_column="city",   # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        geo_column: str = kwargs.get("geo_column")
        if geo_column is None:
            raise ValueError("'geo_column' is required.")
        if geo_column not in self._data_frame.columns:
            raise KeyError(f"Column '{geo_column}' not found in DataFrame.")

        return GeoDistributionCalculator().calculate(
            data_frame=self._data_frame,
            geo_column=geo_column,
            top_n=kwargs.get("top_n", 20),
            secondary_column=kwargs.get("secondary_column"),
        )

class AnalyseGeoClustering(BaseDataAnalysis):
    """Haversine-DBSCAN geographic point clustering.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            lat_column="latitude",
            lon_column="longitude",
            epsilon_km=5.0,
            min_samples=5,
        )
    """

    def analyze(self, **kwargs) -> dict:
        lat_column: str = kwargs.get("lat_column")
        lon_column: str = kwargs.get("lon_column")

        if lat_column is None or lon_column is None:
            raise ValueError("'lat_column' and 'lon_column' are required.")
        for col in (lat_column, lon_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return GeoClusteringCalculator().calculate(
            data_frame=self._data_frame,
            lat_column=lat_column,
            lon_column=lon_column,
            epsilon_km=kwargs.get("epsilon_km", 5.0),
            min_samples=kwargs.get("min_samples", 5),
        )

class AnalyseGeoBoundingBox(BaseDataAnalysis):
    """Bounding box, centroid, diagonal, and dispersion label.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            lat_column="latitude",
            lon_column="longitude",
        )
    """

    def analyze(self, **kwargs) -> dict:
        lat_column: str = kwargs.get("lat_column")
        lon_column: str = kwargs.get("lon_column")

        if lat_column is None or lon_column is None:
            raise ValueError("'lat_column' and 'lon_column' are required.")
        for col in (lat_column, lon_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return GeoBoundingBoxCalculator().calculate(
            data_frame=self._data_frame,
            lat_column=lat_column,
            lon_column=lon_column,
        )

class AnalyseGeoHeatmap(BaseDataAnalysis):
    """Grid-based geographic density heatmap (points per km²).

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            lat_column="latitude",
            lon_column="longitude",
            n_lat_bins=20,
            n_lon_bins=20,
            include_empty_cells=False,
        )
    """

    def analyze(self, **kwargs) -> dict:
        lat_column: str = kwargs.get("lat_column")
        lon_column: str = kwargs.get("lon_column")

        if lat_column is None or lon_column is None:
            raise ValueError("'lat_column' and 'lon_column' are required.")
        for col in (lat_column, lon_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return GeoHeatmapCalculator().calculate(
            data_frame=self._data_frame,
            lat_column=lat_column,
            lon_column=lon_column,
            n_lat_bins=kwargs.get("n_lat_bins", 20),
            n_lon_bins=kwargs.get("n_lon_bins", 20),
            include_empty_cells=kwargs.get("include_empty_cells", False),
        )

class AnalyseProximity(BaseDataAnalysis):
    """Nearest neighbor distances and ANN spatial pattern analysis.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            lat_column="latitude",
            lon_column="longitude",
            include_all_nn=False,
            max_points=2000,
        )
    """

    def analyze(self, **kwargs) -> dict:
        lat_column: str = kwargs.get("lat_column")
        lon_column: str = kwargs.get("lon_column")

        if lat_column is None or lon_column is None:
            raise ValueError("'lat_column' and 'lon_column' are required.")
        for col in (lat_column, lon_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return ProximityAnalysisCalculator().calculate(
            data_frame=self._data_frame,
            lat_column=lat_column,
            lon_column=lon_column,
            include_all_nn=kwargs.get("include_all_nn", False),
            max_points=kwargs.get("max_points", 2_000),
        )

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN 10 — GRAPH ANALYZERS
# ─────────────────────────────────────────────────────────────────────────────

class AnalyseNetworkDensity(BaseDataAnalysis):
    """Graph density, connectivity, and degree distribution.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            source_column="from_node",
            target_column="to_node",
            graph_type="undirected",   # "directed" | "undirected"
            weight_column=None,        # optional
        )
    """

    def analyze(self, **kwargs) -> dict:
        source_column: str = kwargs.get("source_column")
        target_column: str = kwargs.get("target_column")

        if source_column is None or target_column is None:
            raise ValueError("'source_column' and 'target_column' are required.")
        for col in (source_column, target_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return NetworkDensityCalculator().calculate(
            edges=self._data_frame,
            source_column=source_column,
            target_column=target_column,
            graph_type=kwargs.get("graph_type", "undirected"),
            weight_column=kwargs.get("weight_column"),
        )

class AnalyseCentrality(BaseDataAnalysis):
    """Degree, betweenness, closeness, and PageRank centrality.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            source_column="from_node",
            target_column="to_node",
            graph_type="undirected",
            top_n=20,
            damping=0.85,
            weight_column=None,
        )
    """

    def analyze(self, **kwargs) -> dict:
        source_column: str = kwargs.get("source_column")
        target_column: str = kwargs.get("target_column")

        if source_column is None or target_column is None:
            raise ValueError("'source_column' and 'target_column' are required.")
        for col in (source_column, target_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return CentralityAnalysisCalculator().calculate(
            edges=self._data_frame,
            source_column=source_column,
            target_column=target_column,
            graph_type=kwargs.get("graph_type", "undirected"),
            top_n=kwargs.get("top_n", 20),
            damping=kwargs.get("damping", 0.85),
            weight_column=kwargs.get("weight_column"),
        )

class AnalyseCommunityDetection(BaseDataAnalysis):
    """Louvain-style community detection with modularity Q scoring.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            source_column="from_node",
            target_column="to_node",
            random_seed=42,
            weight_column=None,
        )
    """

    def analyze(self, **kwargs) -> dict:
        source_column: str = kwargs.get("source_column")
        target_column: str = kwargs.get("target_column")

        if source_column is None or target_column is None:
            raise ValueError("'source_column' and 'target_column' are required.")
        for col in (source_column, target_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return CommunityDetectionCalculator().calculate(
            edges=self._data_frame,
            source_column=source_column,
            target_column=target_column,
            random_seed=kwargs.get("random_seed", 42),
            weight_column=kwargs.get("weight_column"),
        )

class AnalysePathAnalysis(BaseDataAnalysis):
    """Average path length, diameter, clustering coefficient, small-world σ.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            source_column="from_node",
            target_column="to_node",
            graph_type="undirected",
            weight_column=None,
        )
    """

    def analyze(self, **kwargs) -> dict:
        source_column: str = kwargs.get("source_column")
        target_column: str = kwargs.get("target_column")

        if source_column is None or target_column is None:
            raise ValueError("'source_column' and 'target_column' are required.")
        for col in (source_column, target_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return PathAnalysisCalculator().calculate(
            edges=self._data_frame,
            source_column=source_column,
            target_column=target_column,
            graph_type=kwargs.get("graph_type", "undirected"),
            weight_column=kwargs.get("weight_column"),
        )

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN 11 — SURVIVAL ANALYZERS
# ─────────────────────────────────────────────────────────────────────────────

class AnalyseKaplanMeier(BaseDataAnalysis):
    """Kaplan-Meier survival curve with Greenwood confidence intervals.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            time_column="survival_days",
            event_column="event_occurred",   # 1=event, 0=censored
            confidence_level=0.95,
            group_column=None,               # optional stratification
        )
    """

    def analyze(self, **kwargs) -> dict:
        time_column: str = kwargs.get("time_column")
        event_column: str = kwargs.get("event_column")

        if time_column is None or event_column is None:
            raise ValueError("'time_column' and 'event_column' are required.")
        for col in (time_column, event_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        group_column: str | None = kwargs.get("group_column")
        if group_column is not None and group_column not in self._data_frame.columns:
            raise KeyError(f"Column '{group_column}' not found in DataFrame.")

        return KaplanMeierCalculator().calculate(
            data_frame=self._data_frame,
            time_column=time_column,
            event_column=event_column,
            confidence_level=kwargs.get("confidence_level", 0.95),
            group_column=group_column,
        )

class AnalyseHazardRate(BaseDataAnalysis):
    """Nelson-Aalen cumulative hazard with Gaussian-smoothed instantaneous hazard.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            time_column="survival_days",
            event_column="event_occurred",   # 1=event, 0=censored
            n_smooth_points=100,
        )
    """

    def analyze(self, **kwargs) -> dict:
        time_column: str = kwargs.get("time_column")
        event_column: str = kwargs.get("event_column")

        if time_column is None or event_column is None:
            raise ValueError("'time_column' and 'event_column' are required.")
        for col in (time_column, event_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return HazardRateCalculator().calculate(
            data_frame=self._data_frame,
            time_column=time_column,
            event_column=event_column,
            n_smooth_points=kwargs.get("n_smooth_points", 100),
        )

class AnalyseEventDensity(BaseDataAnalysis):
    """Event frequency, inter-event intervals, burstiness B, and rolling rate.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            event_time_column="failure_time",
            event_indicator_column="is_failure",   # optional: 1=event filter
            window_size=30.0,                      # optional, auto if None
            n_rate_windows=20,
        )
    """

    def analyze(self, **kwargs) -> dict:
        event_time_column: str = kwargs.get("event_time_column")

        if event_time_column is None:
            raise ValueError("'event_time_column' is required.")
        if event_time_column not in self._data_frame.columns:
            raise KeyError(f"Column '{event_time_column}' not found in DataFrame.")

        event_indicator: str | None = kwargs.get("event_indicator_column")
        if event_indicator is not None and event_indicator not in self._data_frame.columns:
            raise KeyError(f"Column '{event_indicator}' not found in DataFrame.")

        return EventDensityCalculator().calculate(
            data_frame=self._data_frame,
            event_time_column=event_time_column,
            event_indicator_column=event_indicator,
            window_size=kwargs.get("window_size"),
            n_rate_windows=kwargs.get("n_rate_windows", 20),
        )

class AnalyseTimeToEvent(BaseDataAnalysis):
    """Time-to-event descriptive statistics, threshold analysis, exponential fit.

    Workflow:
        analyzer = AnalyzerFactory.create("pd.DataFrame", df)
        result = analyzer.analyze(
            time_column="days_to_churn",
            event_column="churned",          # 1=event, 0=censored
            thresholds=[30, 60, 90, 180, 365],
            fit_exponential=True,
        )
    """

    def analyze(self, **kwargs) -> dict:
        time_column: str = kwargs.get("time_column")
        event_column: str = kwargs.get("event_column")

        if time_column is None or event_column is None:
            raise ValueError("'time_column' and 'event_column' are required.")
        for col in (time_column, event_column):
            if col not in self._data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        return TimeToEventCalculator().calculate(
            data_frame=self._data_frame,
            time_column=time_column,
            event_column=event_column,
            thresholds=kwargs.get("thresholds"),
            fit_exponential=kwargs.get("fit_exponential", True),
        )
