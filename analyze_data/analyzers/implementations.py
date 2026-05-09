"""Module for data analysis"""

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
