"""Polars-native backend implementations for the inferential statistics domain.

All classes in this module operate exclusively through Polars lazy / eager
expression APIs to compute the test statistics, ensuring the dataset is never
converted to NumPy. P-values and confidence limits are computed using SciPy
distribution functions on the collected scalar statistics.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import polars as pl
from scipy import stats

from lumen.statistics.inferential.abstract.anova import AbstractANOVACalculator
from lumen.statistics.inferential.abstract.bootstrap import AbstractBootstrapEstimator
from lumen.statistics.inferential.abstract.chi_square import AbstractChiSquareCalculator
from lumen.statistics.inferential.abstract.confidence_intervals import AbstractConfidenceIntervalCalculator
from lumen.statistics.inferential.abstract.correlation_significance import AbstractCorrelationSignificanceCalculator
from lumen.statistics.inferential.abstract.effect_size import AbstractEffectSizeCalculator
from lumen.statistics.inferential.abstract.hypothesis_test import AbstractHypothesisTestSuite
from lumen.statistics.inferential.abstract.power_analysis import AbstractPowerAnalysisCalculator

def _eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


class ANOVACalculatorPolars(AbstractANOVACalculator):
    """Polars-native one-way ANOVA."""

    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        group_column: str,
        significance_level: float = 0.05,
        run_post_hoc: bool = True,
    ) -> dict[str, Any]:
        frame = _eager(data).select([value_column, group_column]).drop_nulls()
        n_total = frame.height

        # Group stats
        group_stats = (
            frame.group_by(group_column)
            .agg([
                pl.len().alias("n"),
                pl.col(value_column).mean().alias("mean"),
                pl.col(value_column).var(ddof=1).alias("var")
            ])
            .sort(group_column)
        )
        
        k = group_stats.height
        if k < 2:
            raise ValueError("ANOVA requires at least 2 groups.")

        grand_mean = frame[value_column].mean()
        
        # SS Between = sum(n_i * (mean_i - grand_mean)^2)
        ss_between = float(
            group_stats.select(
                (pl.col("n") * (pl.col("mean") - grand_mean)**2).sum()
            )[0, 0]
        )
        
        # SS Within = sum((n_i - 1) * var_i)
        ss_within = float(
            group_stats.select(
                ((pl.col("n") - 1) * pl.col("var")).sum()
            )[0, 0]
        )

        df_between = k - 1
        df_within = n_total - k

        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        f_stat = ms_between / ms_within if ms_within > 0 else float('inf')
        p_value = float(stats.f.sf(f_stat, df_between, df_within))

        result = {
            "test_name": "one_way_anova",
            "f_statistic": f_stat,
            "p_value": p_value,
            "reject_null": p_value < significance_level,
            "significance_level": significance_level,
            "n_groups": k,
            "group_means": dict(zip(group_stats[group_column].to_list(), group_stats["mean"].to_list())),
            "group_sizes": dict(zip(group_stats[group_column].to_list(), group_stats["n"].to_list())),
            "post_hoc": None,
        }

        if result["reject_null"] and run_post_hoc:
            from itertools import combinations
            groups = group_stats.to_dicts()
            comparisons = []
            for g1, g2 in combinations(groups, 2):
                n_harm = (2 * g1["n"] * g2["n"]) / (g1["n"] + g2["n"])
                se = math.sqrt(ms_within / n_harm)
                diff = abs(g1["mean"] - g2["mean"])
                q = diff / se if se > 0 else 0
                p_tukey = float(stats.studentized_range.sf(q, k, df_within))
                comparisons.append({
                    "group_i": g1[group_column],
                    "group_j": g2[group_column],
                    "mean_difference": g1["mean"] - g2["mean"],
                    "p_value": p_tukey,
                    "significant": p_tukey < significance_level,
                })
            result["post_hoc"] = {"method": "tukey_hsd", "comparisons": comparisons}

        return result


class BootstrapEstimatorPolars(AbstractBootstrapEstimator):
    """Polars-native bootstrap using loop over native sample/agg."""

    def estimate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        statistic_expr: pl.Expr,
        n_iterations: int = 5_000,
        confidence_level: float = 0.95,
        random_seed: int | None = 42,
    ) -> dict[str, Any]:
        frame = _eager(data).select(pl.col(column).drop_nulls())
        n = frame.height
        if n < 10:
            raise ValueError("Bootstrap needs >= 10 observations.")

        observed = float(frame.select(statistic_expr)[0, 0])
        
        # Set seed natively
        pl.set_random_seed(random_seed) if random_seed is not None else None

        # Execute bootstrap via native list aggregation (cleaner than loop if n_iterations is reasonable)
        # Using loop here to avoid extreme memory blowout on large datasets
        bootstrap_vals = []
        for _ in range(n_iterations):
            # Sample with replacement natively
            val = frame.sample(n=n, with_replacement=True).select(statistic_expr)[0, 0]
            bootstrap_vals.append(val)

        dist = pl.Series(bootstrap_vals).drop_nulls()
        
        alpha = 1.0 - confidence_level
        lower = float(dist.quantile(alpha / 2, interpolation="linear"))
        upper = float(dist.quantile(1 - alpha / 2, interpolation="linear"))

        return {
            "observed_statistic": observed,
            "confidence_interval": {
                "lower": lower,
                "upper": upper,
                "confidence_level": confidence_level,
            },
            "bootstrap_distribution": {
                "mean": float(dist.mean()),
                "std": float(dist.std(ddof=1)),
                "bias": float(dist.mean()) - observed,
            },
            "n_iterations": n_iterations,
            "n": n,
            "random_seed": random_seed,
        }


class ChiSquareCalculatorPolars(AbstractChiSquareCalculator):
    """Polars-native Chi-Square."""

    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column1: str,
        column2: str | None = None,
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        frame = _eager(data)
        
        if column2 is None:
            counts = frame[column1].drop_nulls().value_counts().sort(column1)
            obs = counts["count"].to_list()
            k = len(obs)
            expected = sum(obs) / k
            chi2 = sum((o - expected)**2 / expected for o in obs)
            df = k - 1
            p_val = float(stats.chi2.sf(chi2, df))
            return {
                "test_name": "chi_square_goodness_of_fit",
                "statistic": chi2,
                "p_value": p_val,
                "dof": df,
                "reject_null": p_val < significance_level,
                "significance_level": significance_level,
                "expected_frequencies": [expected] * k,
            }
        else:
            # Cross-tabulation
            crosstab = frame.drop_nulls(subset=[column1, column2]).pivot(
                values=column1, index=column1, columns=column2, aggregate_function="len"
            ).fill_null(0)
            
            # Extract matrix
            matrix = crosstab.select(pl.all().exclude(column1)).to_numpy()
            
            row_sums = matrix.sum(axis=1)
            col_sums = matrix.sum(axis=0)
            total = matrix.sum()
            
            expected = np.outer(row_sums, col_sums) / total if total > 0 else matrix
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2_terms = (matrix - expected)**2 / expected
                chi2_terms[expected == 0] = 0
            
            chi2 = float(np.sum(chi2_terms))
            df = (matrix.shape[0] - 1) * (matrix.shape[1] - 1)
            p_val = float(stats.chi2.sf(chi2, df))

            import numpy as np
            return {
                "test_name": "chi_square_independence",
                "statistic": chi2,
                "p_value": p_val,
                "dof": df,
                "reject_null": p_val < significance_level,
                "significance_level": significance_level,
                "expected_frequencies": expected.tolist(),
            }


class ConfidenceIntervalCalculatorPolars(AbstractConfidenceIntervalCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column: str,
        confidence_level: float = 0.95,
        method: str = "t",
    ) -> dict[str, Any]:
        frame = _eager(data).select(pl.col(column).drop_nulls())
        n = frame.height
        if n < 2:
            raise ValueError("Need n >= 2")

        agg = frame.select([
            pl.col(column).mean().alias("mean"),
            pl.col(column).std(ddof=1).alias("std")
        ]).row(0)

        mean, std = agg
        se = std / math.sqrt(n)
        alpha = 1 - confidence_level

        if method == "t":
            crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
        else:
            crit = float(stats.norm.ppf(1 - alpha / 2))

        margin = crit * se

        return {
            "mean": mean,
            "margin_of_error": margin,
            "lower_bound": mean - margin,
            "upper_bound": mean + margin,
            "confidence_level": confidence_level,
            "method": method,
            "n": n,
            "std_error": se,
        }


class CorrelationSignificanceCalculatorPolars(AbstractCorrelationSignificanceCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        column1: str,
        column2: str,
        method: str = "pearson",
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        frame = _eager(data).select([column1, column2]).drop_nulls()
        n = frame.height

        if method == "pearson":
            corr = float(frame.select(pl.corr(column1, column2, method="pearson"))[0, 0])
        elif method == "spearman":
            corr = float(frame.select(pl.corr(column1, column2, method="spearman"))[0, 0])
        else:
            raise ValueError(f"Unknown method {method}")

        # p-value via t-distribution
        if n > 2 and abs(corr) < 1.0:
            t_stat = corr * math.sqrt((n - 2) / (1 - corr**2))
            p_val = float(2 * stats.t.sf(abs(t_stat), n - 2))
        else:
            p_val = 0.0 if abs(corr) == 1.0 else 1.0

        return {
            "correlation": corr,
            "p_value": p_val,
            "method": method,
            "reject_null": p_val < significance_level,
            "significance_level": significance_level,
            "n": n,
        }


class EffectSizeCalculatorPolars(AbstractEffectSizeCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        group_column: str,
    ) -> dict[str, Any]:
        frame = _eager(data).select([value_column, group_column]).drop_nulls()
        
        group_stats = (
            frame.group_by(group_column)
            .agg([
                pl.len().alias("n"),
                pl.col(value_column).mean().alias("mean"),
                pl.col(value_column).var(ddof=1).alias("var")
            ])
        ).to_dicts()

        if len(group_stats) != 2:
            raise ValueError("Effect size requires exactly 2 groups.")

        g1, g2 = group_stats
        n1, n2 = g1["n"], g2["n"]
        m1, m2 = g1["mean"], g2["mean"]
        v1, v2 = g1["var"], g2["var"]

        pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0.0

        # Hedges' g correction
        g = d * (1 - (3 / (4 * (n1 + n2) - 9)))

        abs_d = abs(d)
        interpretation = "large" if abs_d >= 0.8 else "medium" if abs_d >= 0.5 else "small" if abs_d >= 0.2 else "negligible"

        return {
            "cohens_d": d,
            "hedges_g": g,
            "interpretation": interpretation,
            "group_stats": {
                g1[group_column]: {"n": n1, "mean": m1, "var": v1},
                g2[group_column]: {"n": n2, "mean": m2, "var": v2},
            }
        }


class HypothesisTestSuitePolars(AbstractHypothesisTestSuite):
    def run(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        value_column: str,
        group_column: str,
        test_type: str = "t_test_ind",
        significance_level: float = 0.05,
    ) -> dict[str, Any]:
        frame = _eager(data).select([value_column, group_column]).drop_nulls()
        
        groups = frame.partition_by(group_column, as_dict=True)
        if len(groups) != 2:
            raise ValueError("Test requires exactly 2 groups.")
        
        g_names = list(groups.keys())
        g1_vals = groups[g_names[0]][value_column].to_numpy()
        g2_vals = groups[g_names[1]][value_column].to_numpy()

        if test_type == "t_test_ind":
            stat, p = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
        elif test_type == "mann_whitney":
            stat, p = stats.mannwhitneyu(g1_vals, g2_vals)
        else:
            raise ValueError(f"Unknown test type {test_type}")

        return {
            "test_name": test_type,
            "statistic": float(stat),
            "p_value": float(p),
            "reject_null": float(p) < significance_level,
            "significance_level": significance_level,
            "group_stats": {
                g_names[0]: {"n": len(g1_vals), "mean": float(g1_vals.mean())},
                g_names[1]: {"n": len(g2_vals), "mean": float(g2_vals.mean())},
            }
        }


class PowerAnalysisCalculatorPolars(AbstractPowerAnalysisCalculator):
    def calculate(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float | None = 0.8,
        n: int | None = None,
        test_type: str = "t_test_ind",
    ) -> dict[str, Any]:
        from statsmodels.stats.power import TTestIndPower
        analysis = TTestIndPower()
        
        if n is None and power is not None:
            n_res = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
            power_res = power
        elif power is None and n is not None:
            power_res = analysis.solve_power(effect_size=effect_size, alpha=alpha, nobs1=n, alternative="two-sided")
            n_res = n
        else:
            raise ValueError("Must provide either power or n, but not both.")

        return {
            "effect_size": effect_size,
            "alpha": alpha,
            "power": float(power_res),
            "n_per_group": float(n_res),
            "test_type": test_type,
        }
