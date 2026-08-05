"""Polars-native backend implementations for the descriptive statistics domain.

All classes in this module receive a ``pl.DataFrame | pl.LazyFrame`` and
operate exclusively through Polars lazy / eager expression APIs.
No NumPy, no Pandas, no row-by-row iteration in public methods.
"""
from __future__ import annotations

import math
from typing import Any

import polars as pl

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

def _eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Materialise a LazyFrame; return DataFrame unchanged."""
    return data.collect() if isinstance(data, pl.LazyFrame) else data


def _series(frame: pl.DataFrame, column: str) -> pl.Series:
    """Return the numeric, null-dropped, Float64 series for ``column``."""
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' not found in frame.")
    return frame[column].drop_nulls().cast(pl.Float64)


def _require_min(series: pl.Series, minimum: int, context: str) -> pl.Series:
    if series.len() < minimum:
        raise ValueError(
            f"{context}: need at least {minimum} observations, "
            f"got {series.len()}."
        )
    return series


# ---------------------------------------------------------------------------
# Central Tendency
# ---------------------------------------------------------------------------

class CentralTendencyCalculatorPolars(AbstractCentralTendencyCalculator):
    """Polars-native central tendency calculator.

    Computes mean, median, mode, trimmed mean, and distribution shape hint
    entirely through Polars expression APIs.
    """

    _SYMMETRY_THRESHOLD: float = 0.05

    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        trim_proportion: float = 0.1,
    ) -> dict[str, Any]:
        if not 0.0 <= trim_proportion < 0.5:
            raise ValueError(
                f"trim_proportion must be in [0.0, 0.5). Got {trim_proportion}."
            )

        frame = _eager(data)
        s = _require_min(_series(frame, column), 1, "CentralTendencyCalculator")
        n = s.len()

        mean: float = s.mean()  # type: ignore[assignment]

        # Median via Polars quantile (linear interpolation)
        median: float = s.quantile(0.5, interpolation="linear")  # type: ignore[assignment]

        # Mode — value_counts gives us count per distinct value
        vc = (
            frame.lazy()
            .select(pl.col(column).drop_nulls().cast(pl.Float64))
            .group_by(column)
            .agg(pl.len().alias("_cnt"))
            .sort("_cnt", descending=True)
            .limit(1)
            .collect()
        )
        if vc.is_empty():
            mode: dict[str, Any] = {"value": None, "count": 0}
        else:
            mode = {
                "value": float(vc[column][0]),
                "count": int(vc["_cnt"][0]),
            }

        # Trimmed mean — drop the lowest and highest trim_proportion fraction
        cut_n = int(math.floor(n * trim_proportion))
        trimmed: pl.Series = s.sort()[cut_n: n - cut_n] if cut_n > 0 else s
        trimmed_mean: float = trimmed.mean()  # type: ignore[assignment]

        # Distribution shape hint
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

class DispersionCalculatorPolars(AbstractDispersionCalculator):
    """Polars-native dispersion calculator.

    Computes variance, std, range, IQR, MAD, and coefficient of variation
    using Polars Series / expression APIs only.
    """

    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        ddof: int = 1,
    ) -> dict[str, Any]:
        frame = _eager(data)
        s = _require_min(_series(frame, column), 1, "DispersionCalculator")

        mean: float = s.mean()  # type: ignore[assignment]
        std: float = s.std(ddof=ddof)  # type: ignore[assignment]
        var: float = s.var(ddof=ddof)  # type: ignore[assignment]
        s_min: float = s.min()  # type: ignore[assignment]
        s_max: float = s.max()  # type: ignore[assignment]
        q1: float = s.quantile(0.25, interpolation="linear")  # type: ignore[assignment]
        q3: float = s.quantile(0.75, interpolation="linear")  # type: ignore[assignment]

        # MAD — median of |x - median(x)|
        med: float = s.quantile(0.5, interpolation="linear")  # type: ignore[assignment]
        mad: float = (s - med).abs().quantile(0.5, interpolation="linear")  # type: ignore[assignment]

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

class DistributionClassifierPolars(AbstractDistributionClassifier):
    """Polars-native distribution classifier.

    Fits theoretical distributions using Kolmogorov-Smirnov statistics
    computed entirely from Polars expressions (ECDF vs. theoretical CDF
    via map_elements on sorted arrays — single materialisation only).
    """

    _MINIMUM_SAMPLE_SIZE: int = 8

    # Candidate distributions: (name, requires_positive)
    _CANDIDATES: list[tuple[str, bool]] = [
        ("normal", False),
        ("log_normal", True),
        ("exponential", True),
        ("uniform", False),
        ("gamma", True),
        ("laplace", False),
        ("logistic", False),
    ]

    _TRANSFORM_MAP: dict[str, str] = {
        "log_normal": "log1p",
        "exponential": "log1p",
        "gamma": "sqrt or log1p",
    }

    def classify(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
    ) -> dict[str, Any]:
        frame = _eager(data)
        s = _series(frame, column)
        s = _require_min(s, self._MINIMUM_SAMPLE_SIZE, "DistributionClassifier")

        n = s.len()
        all_positive = s.min() > 0  # type: ignore[operator]
        sorted_s = s.sort()

        # Compute mean, std, skewness (Fisher) natively
        mean: float = s.mean()  # type: ignore[assignment]
        std: float = s.std(ddof=1)  # type: ignore[assignment]
        skewness: float = self._polars_skewness(s, mean, std)
        excess_kurtosis: float = self._polars_excess_kurtosis(s, mean, std)

        fits = self._fit_all(sorted_s, n, all_positive, mean, std)

        if not fits:
            raise RuntimeError("No distributions could be fitted.")

        best = fits[0]
        is_bimodal = self._bimodality_coefficient(skewness, excess_kurtosis, n)
        label = "bimodal" if is_bimodal else best["name"]
        transformation = self._recommend_transform(best["name"], skewness)

        return {
            "best_fit": best,
            "all_fits": [{"name": f["name"], "ks_statistic": f["ks_statistic"], "p_value": f["p_value"]} for f in fits],
            "is_bimodal": is_bimodal,
            "classification_label": label,
            "recommended_transformation": transformation,
            "skewness": skewness,
        }

    # ------------------------------------------------------------------
    # Native stat helpers (no scipy)
    # ------------------------------------------------------------------

    @staticmethod
    def _polars_skewness(s: pl.Series, mean: float, std: float) -> float:
        if std == 0:
            return 0.0
        n = s.len()
        z = (s - mean) / std
        return float((z ** 3).mean() * n * n / ((n - 1) * (n - 2)))

    @staticmethod
    def _polars_excess_kurtosis(s: pl.Series, mean: float, std: float) -> float:
        if std == 0:
            return 0.0
        n = s.len()
        z = (s - mean) / std
        raw = float((z ** 4).mean())
        # Fisher excess kurtosis: ((n+1)k4 - 3(n-1)) * (n-1)/((n-2)(n-3))
        correction = (n - 1) / ((n - 2) * (n - 3)) if n > 3 else 1.0
        return ((n + 1) * raw - 3 * (n - 1)) * correction

    @staticmethod
    def _bimodality_coefficient(skewness: float, excess_kurtosis: float, n: int) -> bool:
        numerator = skewness ** 2 + 1
        denom = excess_kurtosis + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)) if n > 3 else excess_kurtosis + 3
        bc = numerator / denom if denom != 0 else 0.0
        return bc > 0.555

    def _fit_all(
        self,
        sorted_s: pl.Series,
        n: int,
        all_positive: bool,
        mean: float,
        std: float,
    ) -> list[dict[str, Any]]:
        results = []
        for name, needs_positive in self._CANDIDATES:
            if needs_positive and not all_positive:
                continue
            ks = self._ks_statistic(sorted_s, n, name, mean, std)
            if ks is not None:
                results.append({"name": name, "ks_statistic": ks, "p_value": None, "parameters": {}})
        return sorted(results, key=lambda r: r["ks_statistic"])

    @staticmethod
    def _ks_statistic(
        sorted_s: pl.Series,
        n: int,
        dist_name: str,
        mean: float,
        std: float,
    ) -> float | None:
        """Compute one-sample KS statistic using the empirical CDF vs. a theoretical CDF.
        All math is pure Python / Polars — no scipy dependency.
        """
        import math as _math

        try:
            ecdf = pl.Series([(i + 1) / n for i in range(n)])

            if dist_name == "normal":
                # Φ(z) via math.erfc
                theoretical = sorted_s.map_elements(
                    lambda x: 0.5 * _math.erfc(-(x - mean) / (std * _math.sqrt(2))),
                    return_dtype=pl.Float64,
                )
            elif dist_name == "log_normal":
                log_mean = float((sorted_s.log(base=math.e)).mean())  # type: ignore[arg-type]
                log_std = float((sorted_s.log(base=math.e)).std(ddof=1))  # type: ignore[assignment]
                theoretical = sorted_s.map_elements(
                    lambda x: 0.5 * _math.erfc(-((_math.log(x) - log_mean) / (log_std * _math.sqrt(2)))),
                    return_dtype=pl.Float64,
                )
            elif dist_name == "exponential":
                rate = 1.0 / mean if mean > 0 else 1.0
                theoretical = sorted_s.map_elements(
                    lambda x: 1.0 - _math.exp(-rate * x),
                    return_dtype=pl.Float64,
                )
            elif dist_name == "uniform":
                s_min = float(sorted_s[0])
                s_max = float(sorted_s[-1])
                span = s_max - s_min if s_max > s_min else 1.0
                theoretical = sorted_s.map_elements(
                    lambda x: max(0.0, min(1.0, (x - s_min) / span)),
                    return_dtype=pl.Float64,
                )
            elif dist_name == "gamma":
                # Method-of-moments: shape = (mean/std)^2, scale = std^2/mean
                shape = (mean / std) ** 2 if std > 0 else 1.0
                scale = (std ** 2) / mean if mean > 0 else 1.0
                # Regularised lower incomplete gamma via series approximation
                theoretical = sorted_s.map_elements(
                    lambda x: DistributionClassifierPolars._regularised_gamma(shape, x / scale),
                    return_dtype=pl.Float64,
                )
            elif dist_name == "laplace":
                b = std / _math.sqrt(2)
                theoretical = sorted_s.map_elements(
                    lambda x: (
                        0.5 * _math.exp((x - mean) / b)
                        if x < mean
                        else 1.0 - 0.5 * _math.exp(-(x - mean) / b)
                    ),
                    return_dtype=pl.Float64,
                )
            elif dist_name == "logistic":
                s_std = std * _math.pi / _math.sqrt(3)
                theoretical = sorted_s.map_elements(
                    lambda x: 1.0 / (1.0 + _math.exp(-(x - mean) / s_std)),
                    return_dtype=pl.Float64,
                )
            else:
                return None

            diff = (ecdf - theoretical).abs()
            return float(diff.max())  # type: ignore[arg-type]
        except Exception:
            return None

    @staticmethod
    def _regularised_gamma(a: float, x: float) -> float:
        """Regularised lower incomplete gamma P(a, x) via series expansion."""
        if x < 0:
            return 0.0
        if x == 0:
            return 0.0
        import math as _m
        MAX_ITER = 100
        EPS = 1e-10
        term = _m.exp(-x + a * _m.log(x) - _m.lgamma(a + 1))
        total = term
        for k in range(1, MAX_ITER):
            term *= x / (a + k)
            total += term
            if abs(term) < EPS * abs(total):
                break
        return min(1.0, max(0.0, total))

    def _recommend_transform(self, name: str, skewness: float) -> str | None:
        if name in self._TRANSFORM_MAP:
            return self._TRANSFORM_MAP[name]
        if abs(skewness) > 1.0:
            return "box_cox or yeo_johnson"
        return None


# ---------------------------------------------------------------------------
# Frequency Distribution
# ---------------------------------------------------------------------------

class FrequencyDistributionBuilderPolars(AbstractFrequencyDistributionBuilder):
    """Polars-native frequency distribution builder.

    Constructs a histogram-style frequency table using Polars cut / group_by
    expressions with no NumPy dependency.
    """

    def build(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        n_bins: int | None = None,
        bin_method: str = "auto",
    ) -> dict[str, Any]:
        frame = _eager(data)
        s = _require_min(_series(frame, column), 1, "FrequencyDistributionBuilder")
        n = s.len()

        actual_bins = n_bins if n_bins is not None else self._select_bins(s, n, bin_method)
        s_min: float = s.min()  # type: ignore[assignment]
        s_max: float = s.max()  # type: ignore[assignment]

        # Compute bin edges
        step = (s_max - s_min) / actual_bins if s_max != s_min else 1.0
        edges = [s_min + i * step for i in range(actual_bins + 1)]
        edges[-1] = s_max + 1e-10  # inclusive right edge for last bin

        # Use Polars cut to assign bin labels
        labels = [f"[{edges[i]:.4g}, {edges[i+1]:.4g})" for i in range(actual_bins)]
        s_cat = s.cut(breaks=edges[1:-1], labels=labels, left_closed=True)

        # Group-by to get counts
        counts_df = (
            pl.DataFrame({"_val": s, "_bin": s_cat})
            .group_by("_bin")
            .agg(pl.len().alias("_cnt"))
            .sort("_bin")
        )

        total = n
        table = []
        cumulative = 0.0
        for label in labels:
            row = counts_df.filter(pl.col("_bin") == label)
            freq = int(row["_cnt"][0]) if not row.is_empty() else 0
            rel = freq / total
            cumulative += rel
            i = labels.index(label)
            table.append({
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "bin_label": label,
                "frequency": freq,
                "relative_frequency": rel,
                "cumulative_frequency": cumulative,
            })

        return {
            "table": table,
            "n_bins": actual_bins,
            "total_count": total,
            "bin_method": bin_method if n_bins is None else "manual",
        }

    @staticmethod
    def _select_bins(s: pl.Series, n: int, method: str) -> int:
        data_range: float = float(s.max()) - float(s.min())  # type: ignore[operator]
        if method == "sturges" or (method == "auto" and n <= 200):
            return max(1, int(math.ceil(math.log2(n))) + 1)
        if method == "scott":
            h = 3.5 * float(s.std(ddof=1)) / (n ** (1 / 3))  # type: ignore[assignment]
            return max(1, int(math.ceil(data_range / h))) if h > 0 else 10
        # Freedman-Diaconis
        q1: float = s.quantile(0.25, interpolation="linear")  # type: ignore[assignment]
        q3: float = s.quantile(0.75, interpolation="linear")  # type: ignore[assignment]
        iqr = q3 - q1
        h = 2 * iqr / (n ** (1 / 3))
        return max(1, int(math.ceil(data_range / h))) if h > 0 else 10


# ---------------------------------------------------------------------------
# Normality
# ---------------------------------------------------------------------------

class NormalityTestSuitePolars(AbstractNormalityTestSuite):
    """Polars-native normality test suite.

    Implements Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov
    normality tests using native Polars statistics helpers and pure Python
    math.  Results are aggregated via majority vote.
    """

    _MINIMUM_SAMPLE_SIZE: int = 3
    _SW_MAX: int = 5_000

    def run(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        frame = _eager(data)
        s = _series(frame, column)
        s = _require_min(s, self._MINIMUM_SAMPLE_SIZE, "NormalityTestSuite")

        tests = [
            self._shapiro_wilk(s, significance_level),
            self._anderson_darling(s, significance_level),
            self._kolmogorov_smirnov(s, significance_level),
        ]
        normal_votes = sum(1 for t in tests if t["is_normal"])

        return {
            "overall_is_normal": normal_votes >= (len(tests) // 2 + 1),
            "votes_normal": normal_votes,
            "total_tests": len(tests),
            "significance_level": significance_level,
            "tests": tests,
        }

    def _shapiro_wilk(self, s: pl.Series, alpha: float) -> dict[str, Any]:
        """Approximate Shapiro-Wilk W statistic via regression on order statistics."""
        sample = s.sort()
        if s.len() > self._SW_MAX:
            sample = sample[:self._SW_MAX]
            note = f"Sample truncated to {self._SW_MAX} for test validity."
        else:
            note = None

        n = sample.len()
        # W = (Σ aᵢ * x_(i))² / Σ(xᵢ - x̄)²
        # Approximate aᵢ via Blom scores → Φ⁻¹((i - 3/8)/(n + 1/4))
        m_vals = [(self._probit((i + 1 - 0.375) / (n + 0.25))) for i in range(n)]
        norm_m = sum(v ** 2 for v in m_vals) ** 0.5
        a = [v / norm_m for v in m_vals]

        x_vals = sample.to_list()
        w_num = sum(a[i] * x_vals[n - 1 - i] - a[i] * x_vals[i] for i in range(n // 2)) ** 2
        x_mean = float(sample.mean())  # type: ignore[assignment]
        ss = sum((x - x_mean) ** 2 for x in x_vals)
        w_stat = w_num / ss if ss > 0 else 1.0
        w_stat = min(1.0, max(0.0, w_stat))

        # Approximate p-value: small-sample approximation
        is_normal = w_stat > (1.0 - alpha * 0.5)
        return {
            "test_name": "shapiro_wilk",
            "statistic": round(w_stat, 6),
            "p_value": None,
            "is_normal": is_normal,
            "note": note,
        }

    def _anderson_darling(self, s: pl.Series, alpha: float) -> dict[str, Any]:
        """Anderson-Darling A² statistic against N(μ, σ²)."""
        n = s.len()
        mean: float = s.mean()  # type: ignore[assignment]
        std: float = s.std(ddof=1)  # type: ignore[assignment]
        if std == 0:
            return {"test_name": "anderson_darling", "statistic": 0.0, "p_value": None, "is_normal": True, "note": "Zero variance."}

        sorted_s = s.sort()
        x_vals = sorted_s.to_list()

        # Compute Φ((xᵢ - μ) / σ) for each value
        phi = [0.5 * math.erfc(-((x - mean) / (std * math.sqrt(2)))) for x in x_vals]
        phi = [max(1e-15, min(1 - 1e-15, p)) for p in phi]

        a2 = -n - (1.0 / n) * sum(
            (2 * i + 1) * (math.log(phi[i]) + math.log(1 - phi[n - 1 - i]))
            for i in range(n)
        )
        # Adjust for small samples
        a2_adj = a2 * (1 + 4 / n - 25 / (n * n))

        # Critical values at [15%, 10%, 5%, 2.5%, 1%]
        _CRITICAL = {0.15: 0.576, 0.10: 0.656, 0.05: 0.787, 0.025: 0.918, 0.01: 1.092}
        crit = _CRITICAL.get(alpha, 0.787)
        is_normal = a2_adj < crit

        return {
            "test_name": "anderson_darling",
            "statistic": round(a2_adj, 6),
            "p_value": None,
            "is_normal": is_normal,
            "note": f"Critical value at α={alpha}: {crit}",
        }

    def _kolmogorov_smirnov(self, s: pl.Series, alpha: float) -> dict[str, Any]:
        """One-sample KS test against fitted Normal distribution."""
        n = s.len()
        mean: float = s.mean()  # type: ignore[assignment]
        std: float = s.std(ddof=1)  # type: ignore[assignment]
        if std == 0:
            return {"test_name": "kolmogorov_smirnov", "statistic": 0.0, "p_value": 1.0, "is_normal": True, "note": None}

        sorted_s = s.sort()
        ecdf_vals = [(i + 1) / n for i in range(n)]
        cdf_vals = [0.5 * math.erfc(-((x - mean) / (std * math.sqrt(2)))) for x in sorted_s.to_list()]

        ks = max(abs(ecdf_vals[i] - cdf_vals[i]) for i in range(n))
        # Approximate p-value: Kolmogorov distribution
        t = (ks - 0.01) * (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n))
        p_value = max(0.0, min(1.0, 2 * sum(
            ((-1) ** (k - 1)) * math.exp(-2 * k * k * t * t)
            for k in range(1, 50)
        )))

        return {
            "test_name": "kolmogorov_smirnov",
            "statistic": round(ks, 6),
            "p_value": round(p_value, 6),
            "is_normal": p_value > alpha,
            "note": None,
        }

    @staticmethod
    def _probit(p: float) -> float:
        """Rational approximation for Φ⁻¹(p) (Beasley-Springer-Moro)."""
        p = max(1e-15, min(1 - 1e-15, p))
        if p < 0.5:
            t = math.sqrt(-2 * math.log(p))
        else:
            t = math.sqrt(-2 * math.log(1 - p))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        num = c[0] + t * (c[1] + t * c[2])
        den = 1.0 + t * (d[0] + t * (d[1] + t * d[2]))
        x = t - num / den
        return -x if p < 0.5 else x


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

_DEFAULT_PERCENTILES: list[int] = [1, 5, 10, 25, 50, 75, 90, 95, 99]


class PercentilesCalculatorPolars(AbstractPercentilesCalculator):
    """Polars-native percentile calculator with optional outlier detection."""

    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        percentiles: list[int] | None = None,
        outlier_bounds: tuple[int, int] | None = (1, 99),
    ) -> dict[str, Any]:
        frame = _eager(data)
        s = _require_min(_series(frame, column), 1, "PercentilesCalculator")

        pct_list = percentiles if percentiles is not None else _DEFAULT_PERCENTILES
        if any(p < 0 or p > 100 for p in pct_list):
            raise ValueError("All percentile values must be within [0, 100].")

        pct_map = {
            f"p{p}": float(s.quantile(p / 100.0, interpolation="linear"))  # type: ignore[arg-type]
            for p in pct_list
        }

        result: dict[str, Any] = {"percentiles": pct_map, "n": s.len()}

        if outlier_bounds is not None:
            lo, hi = outlier_bounds
            lower_val = float(s.quantile(lo / 100.0, interpolation="linear"))  # type: ignore[arg-type]
            upper_val = float(s.quantile(hi / 100.0, interpolation="linear"))  # type: ignore[arg-type]
            n_below = int((s < lower_val).sum())
            n_above = int((s > upper_val).sum())
            total = s.len()
            result["outlier_detection"] = {
                "n_below_lower_bound": n_below,
                "n_above_upper_bound": n_above,
                "outlier_count": n_below + n_above,
                "outlier_percentage": (n_below + n_above) / total * 100,
                "bounds": {f"p{lo}": lower_val, f"p{hi}": upper_val},
            }

        return result


# ---------------------------------------------------------------------------
# Skewness & Kurtosis
# ---------------------------------------------------------------------------

class SkewnessKurtosisCalculatorPolars(AbstractSkewnessKurtosisCalculator):
    """Polars-native skewness and kurtosis calculator with interpretations."""

    _MINIMUM_SAMPLE_SIZE: int = 4

    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        bias: bool = True,
    ) -> dict[str, Any]:
        frame = _eager(data)
        s = _require_min(_series(frame, column), self._MINIMUM_SAMPLE_SIZE, "SkewnessKurtosisCalculator")

        mean: float = s.mean()  # type: ignore[assignment]
        std: float = s.std(ddof=0 if bias else 1)  # type: ignore[assignment]
        n = s.len()

        if std == 0:
            skewness = 0.0
            excess_kurtosis = 0.0
        else:
            z = (s - mean) / std
            skewness = float((z ** 3).mean())
            excess_kurtosis = float((z ** 4).mean()) - 3.0
            if not bias:
                # Correct for sample bias
                skewness = skewness * n * n / ((n - 1) * (n - 2))
                excess_kurtosis = (
                    (n - 1) * ((n + 1) * excess_kurtosis * n / (n - 1) - 3 * (n - 1))
                    / ((n - 2) * (n - 3))
                )

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

class ValueCountsCalculatorPolars(AbstractValueCountsCalculator):
    """Polars-native value frequency calculator."""

    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        top_n: int | None = None,
        include_missing: bool = True,
    ) -> dict[str, Any]:
        frame = _eager(data)
        if column not in frame.columns:
            raise KeyError(f"Column '{column}' not found in frame.")

        col = frame[column]
        n_total = col.len()
        n_missing = col.null_count()
        n_valid = n_total - n_missing

        # Count distinct values (including or excluding nulls)
        working = col if include_missing else col.drop_nulls()
        n_unique = int(working.drop_nulls().n_unique())

        counts_df = (
            pl.DataFrame({column: working})
            .group_by(column)
            .agg(pl.len().alias("_count"))
            .sort("_count", descending=True)
        )

        if top_n is not None:
            counts_df = counts_df.head(top_n)

        table = [
            {
                "value": str(counts_df[column][i]),
                "frequency": int(counts_df["_count"][i]),
                "relative_frequency": float(counts_df["_count"][i]) / n_total,
                "percentage": float(counts_df["_count"][i]) / n_total * 100,
            }
            for i in range(counts_df.height)
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
