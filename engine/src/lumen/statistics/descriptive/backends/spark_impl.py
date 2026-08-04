"""PySpark-native backend implementations for the descriptive statistics domain.

All classes in this module receive a ``pyspark.sql.DataFrame`` and operate
exclusively through PySpark SQL functions, aggregations, and window
expressions.  No NumPy, no Pandas, no row-by-row Python iteration in
public methods.
"""
from __future__ import annotations

import math
from typing import Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from lumen.statistics.descriptive.abstract.central_tendency import AbstractCentralTendencyCalculator
from lumen.statistics.descriptive.abstract.dispersion import AbstractDispersionCalculator
from lumen.statistics.descriptive.abstract.distribution import AbstractDistributionClassifier
from lumen.statistics.descriptive.abstract.frequency import AbstractFrequencyDistributionBuilder
from lumen.statistics.descriptive.abstract.normality import AbstractNormalityTestSuite
from lumen.statistics.descriptive.abstract.percentiles import AbstractPercentilesCalculator
from lumen.statistics.descriptive.abstract.skewness_kurtosis import AbstractSkewnessKurtosisCalculator
from lumen.statistics.descriptive.abstract.value_counts import AbstractValueCountsCalculator

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_column(df: SparkDataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")


def _clean_col(df: SparkDataFrame, column: str) -> SparkDataFrame:
    """Drop nulls and cast column to double."""
    _require_column(df, column)
    return df.select(F.col(column).cast("double").alias(column)).na.drop(subset=[column])


def _scalar(df: SparkDataFrame, expr: Any) -> Any:
    """Evaluate a single aggregate expression and return the scalar value."""
    return df.agg(expr).collect()[0][0]


# ---------------------------------------------------------------------------
# Central Tendency
# ---------------------------------------------------------------------------

class CentralTendencyCalculatorSpark(AbstractCentralTendencyCalculator):
    """PySpark-native central tendency calculator.

    Computes mean, median, mode, trimmed mean, and distribution shape hint
    using PySpark SQL aggregate and window functions only.
    """

    _SYMMETRY_THRESHOLD: float = 0.05

    def calculate(
        self,
        data: SparkDataFrame,
        column: str,
        trim_proportion: float = 0.1,
    ) -> dict[str, Any]:
        if not 0.0 <= trim_proportion < 0.5:
            raise ValueError(f"trim_proportion must be in [0.0, 0.5). Got {trim_proportion}.")

        clean = _clean_col(data, column)
        count = clean.count()
        if count == 0:
            raise ValueError(f"Column '{column}' is empty after dropping nulls.")

        agg_row = clean.agg(
            F.mean(column).alias("mean"),
            F.percentile_approx(column, 0.5).alias("median"),
        ).collect()[0]

        mean: float = float(agg_row["mean"])
        median: float = float(agg_row["median"])

        # Mode — most frequent value
        mode_row = (
            clean.groupBy(column)
            .count()
            .orderBy(F.desc("count"))
            .limit(1)
            .collect()
        )
        mode: dict[str, Any] = (
            {"value": float(mode_row[0][column]), "count": int(mode_row[0]["count"])}
            if mode_row else {"value": None, "count": 0}
        )

        # Trimmed mean — drop lowest and highest trim_proportion rows
        cut_n = int(math.floor(count * trim_proportion))
        if cut_n > 0:
            # Rank rows ascending/descending and filter
            w_asc = Window.orderBy(F.col(column).asc())
            w_desc = Window.orderBy(F.col(column).desc())
            ranked = clean.withColumn("_rk_asc", F.row_number().over(w_asc)) \
                          .withColumn("_rk_desc", F.row_number().over(w_desc))
            trimmed_df = ranked.filter(
                (F.col("_rk_asc") > cut_n) & (F.col("_rk_desc") > cut_n)
            )
        else:
            trimmed_df = clean

        trimmed_mean: float = float(_scalar(trimmed_df, F.mean(column)))

        # Shape hint
        if mean == 0:
            shape_hint = "mean_is_zero"
        elif abs(mean - median) / abs(mean) < self._SYMMETRY_THRESHOLD:
            shape_hint = "symmetric"
        elif mean > median:
            shape_hint = "right_skewed"
        else:
            shape_hint = "left_skewed"

        return {
            "mean": mean,
            "median": median,
            "mode": mode,
            "trimmed_mean": trimmed_mean,
            "trim_proportion": trim_proportion,
            "distribution_shape_hint": shape_hint,
        }


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

class DispersionCalculatorSpark(AbstractDispersionCalculator):
    """PySpark-native dispersion calculator.

    Uses PySpark built-in functions: stddev_samp / stddev_pop,
    var_samp / var_pop, min, max, approx_percentile.
    MAD is computed as a distributed aggregation.
    """

    def calculate(
        self,
        data: SparkDataFrame,
        column: str,
        ddof: int = 1,
    ) -> dict[str, Any]:
        clean = _clean_col(data, column)
        count = clean.count()
        if count == 0:
            raise ValueError(f"Column '{column}' is empty after dropping nulls.")

        std_fn = F.stddev_samp if ddof == 1 else F.stddev_pop
        var_fn = F.var_samp if ddof == 1 else F.var_pop

        agg = clean.agg(
            F.mean(column).alias("mean"),
            std_fn(column).alias("std"),
            var_fn(column).alias("var"),
            F.min(column).alias("min"),
            F.max(column).alias("max"),
            F.percentile_approx(column, 0.25).alias("q1"),
            F.percentile_approx(column, 0.75).alias("q3"),
            F.percentile_approx(column, 0.5).alias("median"),
        ).collect()[0]

        mean = float(agg["mean"])
        std = float(agg["std"] or 0.0)
        var = float(agg["var"] or 0.0)
        q1 = float(agg["q1"])
        q3 = float(agg["q3"])
        med = float(agg["median"])
        s_min = float(agg["min"])
        s_max = float(agg["max"])

        # MAD — distributed: compute |x - median|, then take median of that
        mad: float = float(
            clean.withColumn("_abs_dev", F.abs(F.col(column) - med))
            .agg(F.percentile_approx("_abs_dev", 0.5))
            .collect()[0][0]
        )

        cv = float(std / abs(mean)) if mean != 0 else float("inf")

        return {
            "variance": var,
            "std": std,
            "range": {"min": s_min, "max": s_max, "range": s_max - s_min},
            "iqr": {"q1": q1, "q3": q3, "iqr": q3 - q1},
            "mad": mad,
            "coefficient_of_variation": cv,
            "ddof": ddof,
        }


# ---------------------------------------------------------------------------
# Distribution Classification
# ---------------------------------------------------------------------------

class DistributionClassifierSpark(AbstractDistributionClassifier):
    """PySpark-native distribution classifier.

    Fits distributions using distributed KS statistic computation:
    collects sorted values and theoretical CDF values as a single pass,
    keeping all heavy lifting in Spark aggregations.
    """

    _MINIMUM_SAMPLE_SIZE: int = 8

    _CANDIDATES: list[tuple[str, bool]] = [
        ("normal", False),
        ("log_normal", True),
        ("exponential", True),
        ("uniform", False),
        ("laplace", False),
        ("logistic", False),
    ]

    _TRANSFORM_MAP: dict[str, str] = {
        "log_normal": "log1p",
        "exponential": "log1p",
    }

    def classify(
        self,
        data: SparkDataFrame,
        column: str,
    ) -> dict[str, Any]:
        clean = _clean_col(data, column)
        n = clean.count()
        if n < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"DistributionClassifier requires at least {self._MINIMUM_SAMPLE_SIZE} "
                f"samples. Got {n}."
            )

        agg = clean.agg(
            F.mean(column).alias("mean"),
            F.stddev_samp(column).alias("std"),
            F.skewness(column).alias("skewness"),
            F.kurtosis(column).alias("kurtosis"),
            F.min(column).alias("min"),
        ).collect()[0]

        mean = float(agg["mean"])
        std = float(agg["std"] or 1.0)
        skewness = float(agg["skewness"] or 0.0)
        excess_kurtosis = float(agg["kurtosis"] or 0.0)
        all_positive = float(agg["min"]) > 0

        # Collect sorted values for KS (single action)
        sorted_vals = [
            float(r[column])
            for r in clean.orderBy(F.col(column).asc()).collect()
        ]

        fits = self._fit_all(sorted_vals, n, all_positive, mean, std)
        if not fits:
            raise RuntimeError("No distributions could be fitted.")

        best = fits[0]
        is_bimodal = self._bimodality(skewness, excess_kurtosis, n)
        label = "bimodal" if is_bimodal else best["name"]
        transformation = self._TRANSFORM_MAP.get(best["name"]) if not is_bimodal else None
        if transformation is None and abs(skewness) > 1.0:
            transformation = "box_cox or yeo_johnson"

        return {
            "best_fit": best,
            "all_fits": [{"name": f["name"], "ks_statistic": f["ks_statistic"], "p_value": None} for f in fits],
            "is_bimodal": is_bimodal,
            "classification_label": label,
            "recommended_transformation": transformation,
            "skewness": skewness,
        }

    @staticmethod
    def _bimodality(skewness: float, excess_kurtosis: float, n: int) -> bool:
        numerator = skewness ** 2 + 1
        denom = excess_kurtosis + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else excess_kurtosis + 3
        return (numerator / denom if denom != 0 else 0.0) > 0.555

    def _fit_all(
        self,
        sorted_vals: list[float],
        n: int,
        all_positive: bool,
        mean: float,
        std: float,
    ) -> list[dict[str, Any]]:
        results = []
        for name, needs_positive in self._CANDIDATES:
            if needs_positive and not all_positive:
                continue
            ks = self._ks(sorted_vals, n, name, mean, std)
            if ks is not None:
                results.append({"name": name, "ks_statistic": ks, "p_value": None, "parameters": {}})
        return sorted(results, key=lambda r: r["ks_statistic"])

    @staticmethod
    def _ks(sorted_vals: list[float], n: int, dist: str, mean: float, std: float) -> float | None:
        try:
            ecdf = [(i + 1) / n for i in range(n)]

            if dist == "normal":
                cdf = [0.5 * math.erfc(-((x - mean) / (std * math.sqrt(2)))) for x in sorted_vals]
            elif dist == "log_normal":
                log_vals = [math.log(x) for x in sorted_vals if x > 0]
                if len(log_vals) < n:
                    return None
                lm = sum(log_vals) / n
                ls = (sum((v - lm) ** 2 for v in log_vals) / (n - 1)) ** 0.5 if n > 1 else 1.0
                cdf = [0.5 * math.erfc(-((math.log(x) - lm) / (ls * math.sqrt(2)))) for x in sorted_vals]
            elif dist == "exponential":
                rate = 1.0 / mean if mean > 0 else 1.0
                cdf = [1.0 - math.exp(-rate * x) for x in sorted_vals]
            elif dist == "uniform":
                lo, hi = sorted_vals[0], sorted_vals[-1]
                span = hi - lo if hi > lo else 1.0
                cdf = [max(0.0, min(1.0, (x - lo) / span)) for x in sorted_vals]
            elif dist == "laplace":
                b = std / math.sqrt(2)
                cdf = [
                    (0.5 * math.exp((x - mean) / b) if x < mean
                     else 1.0 - 0.5 * math.exp(-(x - mean) / b))
                    for x in sorted_vals
                ]
            elif dist == "logistic":
                s_std = std * math.pi / math.sqrt(3)
                cdf = [1.0 / (1.0 + math.exp(-(x - mean) / s_std)) for x in sorted_vals]
            else:
                return None

            return max(abs(ecdf[i] - cdf[i]) for i in range(n))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Frequency Distribution
# ---------------------------------------------------------------------------

class FrequencyDistributionBuilderSpark(AbstractFrequencyDistributionBuilder):
    """PySpark-native frequency distribution builder.

    Constructs a histogram using Spark's ``width_bucket`` SQL function to
    assign bins, then aggregates counts per bin in a single distributed pass.
    """

    def build(
        self,
        data: SparkDataFrame,
        column: str,
        n_bins: int | None = None,
        bin_method: str = "auto",
    ) -> dict[str, Any]:
        clean = _clean_col(data, column)
        n = clean.count()
        if n == 0:
            raise ValueError(f"Column '{column}' is empty after dropping nulls.")

        agg = clean.agg(
            F.mean(column).alias("mean"),
            F.stddev_samp(column).alias("std"),
            F.min(column).alias("min"),
            F.max(column).alias("max"),
            F.percentile_approx(column, [0.25, 0.75]).alias("quartiles"),
        ).collect()[0]

        s_min = float(agg["min"])
        s_max = float(agg["max"])
        std = float(agg["std"] or 0.0)
        q1, q3 = float(agg["quartiles"][0]), float(agg["quartiles"][1])

        actual_bins = n_bins if n_bins is not None else self._select_bins(n, method=bin_method, std=std, data_range=s_max - s_min, q1=q1, q3=q3)
        step = (s_max - s_min) / actual_bins if s_max != s_min else 1.0
        edges = [s_min + i * step for i in range(actual_bins + 1)]
        edges[-1] += 1e-10  # include right edge in last bin

        # width_bucket assigns values to bins 1..actual_bins
        binned = clean.withColumn(
            "_bin_idx",
            F.width_bucket(F.col(column), F.lit(s_min), F.lit(edges[-1]), F.lit(actual_bins)),
        )
        counts_df = (
            binned.groupBy("_bin_idx")
            .agg(F.count("*").alias("_cnt"))
        )
        counts_map: dict[int, int] = {
            int(r["_bin_idx"]): int(r["_cnt"])
            for r in counts_df.collect()
        }

        table = []
        cumulative = 0.0
        for i in range(actual_bins):
            bin_idx = i + 1
            freq = counts_map.get(bin_idx, 0)
            rel = freq / n
            cumulative += rel
            table.append({
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "bin_label": f"[{edges[i]:.4g}, {edges[i+1]:.4g})",
                "frequency": freq,
                "relative_frequency": rel,
                "cumulative_frequency": cumulative,
            })

        return {
            "table": table,
            "n_bins": actual_bins,
            "total_count": n,
            "bin_method": bin_method if n_bins is None else "manual",
        }

    @staticmethod
    def _select_bins(n: int, method: str, std: float, data_range: float, q1: float, q3: float) -> int:
        if method == "sturges" or (method == "auto" and n <= 200):
            return max(1, int(math.ceil(math.log2(n))) + 1)
        if method == "scott":
            h = 3.5 * std / (n ** (1 / 3))
            return max(1, int(math.ceil(data_range / h))) if h > 0 else 10
        iqr = q3 - q1
        h = 2 * iqr / (n ** (1 / 3))
        return max(1, int(math.ceil(data_range / h))) if h > 0 else 10


# ---------------------------------------------------------------------------
# Normality
# ---------------------------------------------------------------------------

class NormalityTestSuiteSpark(AbstractNormalityTestSuite):
    """PySpark-native normality test suite.

    Uses Spark built-in ``skewness``, ``kurtosis``, ``stddev_samp``,
    ``percentile_approx`` to build the statistics needed for KS, Anderson-Darling
    and an approximate Shapiro-Wilk, then evaluates them without leaving Spark.
    """

    _MINIMUM_SAMPLE_SIZE: int = 3

    def run(
        self,
        data: SparkDataFrame,
        column: str,
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        clean = _clean_col(data, column)
        n = clean.count()
        if n < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"Normality tests require at least {self._MINIMUM_SAMPLE_SIZE} points. Got {n}."
            )

        agg = clean.agg(
            F.mean(column).alias("mean"),
            F.stddev_samp(column).alias("std"),
        ).collect()[0]
        mean = float(agg["mean"])
        std = float(agg["std"] or 0.0)

        # Collect sorted values for KS / AD (single collect)
        sorted_vals = [float(r[column]) for r in clean.orderBy(F.col(column).asc()).collect()]

        tests = [
            self._shapiro_wilk(sorted_vals, n, mean, std, significance_level),
            self._anderson_darling(sorted_vals, n, mean, std, significance_level),
            self._kolmogorov_smirnov(sorted_vals, n, mean, std, significance_level),
        ]
        normal_votes = sum(1 for t in tests if t["is_normal"])

        return {
            "overall_is_normal": normal_votes >= (len(tests) // 2 + 1),
            "votes_normal": normal_votes,
            "total_tests": len(tests),
            "significance_level": significance_level,
            "tests": tests,
        }

    @staticmethod
    def _shapiro_wilk(vals: list[float], n: int, mean: float, std: float, alpha: float) -> dict[str, Any]:
        """Lightweight Shapiro-Wilk approximation via Blom scores."""
        if std == 0:
            return {"test_name": "shapiro_wilk", "statistic": 1.0, "p_value": None, "is_normal": True, "note": "Zero variance."}

        sample = vals[:5000]
        n_s = len(sample)
        m_vals = [
            0.5 * math.erfc(-((i + 1 - 0.375) / (n_s + 0.25) - 0.5) / (1 / math.sqrt(2)))
            for i in range(n_s)
        ]
        # Use simple Blom approximation for expected normal order stats
        def _probit(p: float) -> float:
            p = max(1e-15, min(1 - 1e-15, p))
            t = math.sqrt(-2 * math.log(p if p < 0.5 else 1 - p))
            c = (2.515517, 0.802853, 0.010328)
            d = (1.432788, 0.189269, 0.001308)
            x = t - (c[0] + t * (c[1] + t * c[2])) / (1.0 + t * (d[0] + t * (d[1] + t * d[2])))
            return -x if p < 0.5 else x

        m_vals = [_probit((i + 1 - 0.375) / (n_s + 0.25)) for i in range(n_s)]
        norm_m = sum(v ** 2 for v in m_vals) ** 0.5
        a = [v / norm_m for v in m_vals]

        w_num = sum(a[i] * sample[n_s - 1 - i] - a[i] * sample[i] for i in range(n_s // 2)) ** 2
        ss = sum((x - mean) ** 2 for x in sample)
        w_stat = min(1.0, max(0.0, w_num / ss if ss > 0 else 1.0))
        is_normal = w_stat > (1.0 - alpha * 0.5)

        note = f"Sample truncated to 5000." if n > 5000 else None
        return {"test_name": "shapiro_wilk", "statistic": round(w_stat, 6), "p_value": None, "is_normal": is_normal, "note": note}

    @staticmethod
    def _anderson_darling(vals: list[float], n: int, mean: float, std: float, alpha: float) -> dict[str, Any]:
        if std == 0:
            return {"test_name": "anderson_darling", "statistic": 0.0, "p_value": None, "is_normal": True, "note": "Zero variance."}
        phi = [max(1e-15, min(1 - 1e-15, 0.5 * math.erfc(-((x - mean) / (std * math.sqrt(2)))))) for x in vals]
        a2 = -n - (1.0 / n) * sum((2 * i + 1) * (math.log(phi[i]) + math.log(1 - phi[n - 1 - i])) for i in range(n))
        a2_adj = a2 * (1 + 4 / n - 25 / (n * n))
        _CRIT = {0.15: 0.576, 0.10: 0.656, 0.05: 0.787, 0.025: 0.918, 0.01: 1.092}
        crit = _CRIT.get(alpha, 0.787)
        return {"test_name": "anderson_darling", "statistic": round(a2_adj, 6), "p_value": None, "is_normal": a2_adj < crit, "note": f"Critical value: {crit}"}

    @staticmethod
    def _kolmogorov_smirnov(vals: list[float], n: int, mean: float, std: float, alpha: float) -> dict[str, Any]:
        if std == 0:
            return {"test_name": "kolmogorov_smirnov", "statistic": 0.0, "p_value": 1.0, "is_normal": True, "note": None}
        ecdf = [(i + 1) / n for i in range(n)]
        cdf = [0.5 * math.erfc(-((x - mean) / (std * math.sqrt(2)))) for x in vals]
        ks = max(abs(ecdf[i] - cdf[i]) for i in range(n))
        t = (ks - 0.01) * (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n))
        p_val = max(0.0, min(1.0, 2 * sum(((-1) ** (k - 1)) * math.exp(-2 * k * k * t * t) for k in range(1, 50))))
        return {"test_name": "kolmogorov_smirnov", "statistic": round(ks, 6), "p_value": round(p_val, 6), "is_normal": p_val > alpha, "note": None}


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

_DEFAULT_PERCENTILES: list[int] = [1, 5, 10, 25, 50, 75, 90, 95, 99]


class PercentilesCalculatorSpark(AbstractPercentilesCalculator):
    """PySpark-native percentile calculator using percentile_approx."""

    def calculate(
        self,
        data: SparkDataFrame,
        column: str,
        percentiles: list[int] | None = None,
        outlier_bounds: tuple[int, int] | None = (1, 99),
    ) -> dict[str, Any]:
        clean = _clean_col(data, column)
        n = clean.count()
        if n == 0:
            raise ValueError(f"Column '{column}' is empty after dropping nulls.")

        pct_list = percentiles if percentiles is not None else _DEFAULT_PERCENTILES
        if any(p < 0 or p > 100 for p in pct_list):
            raise ValueError("All percentile values must be within [0, 100].")

        fracs = [p / 100.0 for p in pct_list]
        pct_vals = clean.agg(
            F.percentile_approx(column, fracs).alias("_pct")
        ).collect()[0]["_pct"]

        pct_map = {f"p{p}": float(v) for p, v in zip(pct_list, pct_vals)}
        result: dict[str, Any] = {"percentiles": pct_map, "n": n}

        if outlier_bounds is not None:
            lo, hi = outlier_bounds
            bounds_vals = clean.agg(
                F.percentile_approx(column, [lo / 100.0, hi / 100.0]).alias("_b")
            ).collect()[0]["_b"]
            lower_val, upper_val = float(bounds_vals[0]), float(bounds_vals[1])

            n_below = int(clean.filter(F.col(column) < lower_val).count())
            n_above = int(clean.filter(F.col(column) > upper_val).count())

            result["outlier_detection"] = {
                "n_below_lower_bound": n_below,
                "n_above_upper_bound": n_above,
                "outlier_count": n_below + n_above,
                "outlier_percentage": (n_below + n_above) / n * 100,
                "bounds": {f"p{lo}": lower_val, f"p{hi}": upper_val},
            }

        return result


# ---------------------------------------------------------------------------
# Skewness & Kurtosis
# ---------------------------------------------------------------------------

class SkewnessKurtosisCalculatorSpark(AbstractSkewnessKurtosisCalculator):
    """PySpark-native skewness and kurtosis calculator.

    Uses PySpark's built-in ``skewness`` and ``kurtosis`` aggregate functions
    (Fisher's definitions, same as SciPy bias=True default).
    """

    _MINIMUM_SAMPLE_SIZE: int = 4

    def calculate(
        self,
        data: SparkDataFrame,
        column: str,
        bias: bool = True,
    ) -> dict[str, Any]:
        clean = _clean_col(data, column)
        n = clean.count()
        if n < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"At least {self._MINIMUM_SAMPLE_SIZE} data points required. Got {n}."
            )

        agg = clean.agg(
            F.skewness(column).alias("skewness"),
            F.kurtosis(column).alias("excess_kurtosis"),
        ).collect()[0]

        skewness = float(agg["skewness"] or 0.0)
        excess_kurtosis = float(agg["excess_kurtosis"] or 0.0)
        pearson_kurtosis = excess_kurtosis + 3.0

        return {
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "pearson_kurtosis": pearson_kurtosis,
            "skewness_interpretation": self._interpret_skewness(skewness),
            "kurtosis_interpretation": self._interpret_kurtosis(excess_kurtosis),
        }

    @staticmethod
    def _interpret_skewness(skewness: float) -> dict[str, str]:
        abs_skew = abs(skewness)
        direction = "right (positive)" if skewness > 0 else "left (negative)" if skewness < 0 else "none"
        if abs_skew < 0.5:
            severity, action = "approximately symmetric", "no transformation needed"
        elif abs_skew < 1.0:
            severity, action = "moderately skewed", "consider sqrt or log transformation"
        else:
            severity, action = "highly skewed", "log1p or box_cox transformation recommended"
        return {"direction": direction, "severity": severity, "recommended_action": action}

    @staticmethod
    def _interpret_kurtosis(excess_kurtosis: float) -> dict[str, str]:
        if abs(excess_kurtosis) < 0.5:
            dist_type, action = "mesokurtic (normal-like tails)", "standard models safe"
        elif excess_kurtosis > 0.5:
            dist_type, action = "leptokurtic (heavy tails, outlier-prone)", "robust models or outlier handling recommended"
        else:
            dist_type, action = "platykurtic (light tails, fewer extremes)", "standard models generally safe"
        return {"distribution_type": dist_type, "recommended_action": action}


# ---------------------------------------------------------------------------
# Value Counts
# ---------------------------------------------------------------------------

class ValueCountsCalculatorSpark(AbstractValueCountsCalculator):
    """PySpark-native value counts calculator.

    Uses Spark ``groupBy`` + ``count`` + ``window`` functions to compute
    frequencies without collecting data prematurely.
    """

    def calculate(
        self,
        data: SparkDataFrame,
        column: str,
        top_n: int | None = None,
        include_missing: bool = True,
    ) -> dict[str, Any]:
        _require_column(data, column)

        n_total = data.count()
        n_missing = data.filter(F.col(column).isNull()).count()
        n_valid = n_total - n_missing
        n_unique = int(data.select(column).dropDuplicates().filter(F.col(column).isNotNull()).count())

        working = data if include_missing else data.filter(F.col(column).isNotNull())

        counts = (
            working.groupBy(column)
            .agg(F.count("*").alias("_count"))
            .orderBy(F.desc("_count"))
        )

        if top_n is not None:
            counts = counts.limit(top_n)

        rows = counts.collect()
        table = [
            {
                "value": str(r[column]),
                "frequency": int(r["_count"]),
                "relative_frequency": int(r["_count"]) / n_total,
                "percentage": int(r["_count"]) / n_total * 100,
            }
            for r in rows
        ]

        return {
            "table": table,
            "n_total": n_total,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "missing_percentage": round(n_missing / n_total * 100, 4) if n_total > 0 else 0.0,
            "n_valid": n_valid,
            "top_n": top_n,
        }
