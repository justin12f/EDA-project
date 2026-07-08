"""Polars-native backend implementations for the relational statistics domain."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression

from relational.abstract.contingency_analysis import AbstractContingencyAnalysisCalculator
from relational.abstract.correlation_matrix import AbstractCorrelationMatrixCalculator
from relational.abstract.cross_correlation import AbstractCrossCorrelationCalculator
from relational.abstract.granger_causality import AbstractGrangerCausalityCalculator
from relational.abstract.interaction_effects import AbstractInteractionEffectsCalculator
from relational.abstract.multicollinearity import AbstractMulticollinearityCalculator
from relational.abstract.mutual_information import AbstractMutualInformationCalculator
from relational.abstract.partial_correlation import AbstractPartialCorrelationCalculator

def _eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


class ContingencyAnalysisCalculatorPolars(AbstractContingencyAnalysisCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        col1: str,
        col2: str,
    ) -> dict[str, Any]:
        frame = _eager(data).drop_nulls(subset=[col1, col2])
        
        crosstab = frame.pivot(
            values=col1, index=col1, columns=col2, aggregate_function="len"
        ).fill_null(0)
        
        matrix = crosstab.select(pl.all().exclude(col1)).to_numpy()
        
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)
        total = matrix.sum()
        
        expected = np.outer(row_sums, col_sums) / total if total > 0 else matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_terms = (matrix - expected)**2 / expected
            chi2_terms[expected == 0] = 0
            
        chi2 = float(np.sum(chi2_terms))
        dof = (matrix.shape[0] - 1) * (matrix.shape[1] - 1)
        p_val = float(stats.chi2.sf(chi2, dof))
        
        min_dim = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        cramers_v = float(math.sqrt(chi2 / (total * min_dim))) if total * min_dim > 0 else 0.0

        return {
            "chi_square": chi2,
            "p_value": p_val,
            "dof": dof,
            "cramers_v": cramers_v,
            "contingency_table": matrix.tolist(),
        }


class CorrelationMatrixCalculatorPolars(AbstractCorrelationMatrixCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        columns: list[str],
        method: str = "pearson",
    ) -> dict[str, Any]:
        frame = _eager(data).select(columns).drop_nulls()
        
        if method not in ["pearson", "spearman"]:
            raise ValueError(f"Unknown method {method}")
            
        # Polars native correlation matrix
        corr_matrix = []
        p_matrix = []
        n = frame.height
        
        for c1 in columns:
            c_row = []
            p_row = []
            for c2 in columns:
                if c1 == c2:
                    c_row.append(1.0)
                    p_row.append(0.0)
                else:
                    corr = float(frame.select(pl.corr(c1, c2, method=method))[0, 0] or 0.0)
                    c_row.append(corr)
                    
                    if n > 2 and abs(corr) < 1.0:
                        t_stat = corr * math.sqrt((n - 2) / (1 - corr**2))
                        p_val = float(2 * stats.t.sf(abs(t_stat), n - 2))
                    else:
                        p_val = 0.0 if abs(corr) == 1.0 else 1.0
                    p_row.append(p_val)
                    
            corr_matrix.append(c_row)
            p_matrix.append(p_row)

        return {
            "correlation_matrix": corr_matrix,
            "p_value_matrix": p_matrix,
            "columns": columns,
            "method": method,
            "n_observations": n,
        }


class CrossCorrelationCalculatorPolars(AbstractCrossCorrelationCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        col1: str,
        col2: str,
        max_lag: int = 10,
    ) -> dict[str, Any]:
        frame = _eager(data).select([col1, col2])
        
        lags = range(-max_lag, max_lag + 1)
        res = []
        
        for lag in lags:
            if lag < 0:
                df_shifted = frame.with_columns(pl.col(col2).shift(-lag).alias(f"shifted_{col2}"))
            else:
                df_shifted = frame.with_columns(pl.col(col1).shift(lag).alias(f"shifted_{col1}"))
                
            c1_used = col1 if lag < 0 else f"shifted_{col1}"
            c2_used = f"shifted_{col2}" if lag < 0 else col2
            
            corr = float(df_shifted.select(pl.corr(c1_used, c2_used))[0, 0] or 0.0)
            res.append({"lag": lag, "correlation": corr})

        return {"cross_correlations": res, "max_lag": max_lag}


class GrangerCausalityCalculatorPolars(AbstractGrangerCausalityCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target_column: str,
        predictor_column: str,
        max_lag: int = 5,
    ) -> dict[str, Any]:
        from statsmodels.tsa.stattools import grangercausalitytests
        
        frame = _eager(data).select([target_column, predictor_column]).drop_nulls()
        arr = frame.to_numpy()
        
        if len(arr) <= max_lag:
            raise ValueError("Not enough observations for max_lag.")
            
        gc_res = grangercausalitytests(arr, maxlag=max_lag, verbose=False)
        
        results = {}
        for lag, tests in gc_res.items():
            f_test = tests[0]['ssr_ftest']
            results[f"lag_{lag}"] = {
                "f_statistic": float(f_test[0]),
                "p_value": float(f_test[1]),
                "df_num": int(f_test[2]),
                "df_denom": int(f_test[3]),
            }

        return {"granger_causality": results, "max_lag": max_lag}


class InteractionEffectsCalculatorPolars(AbstractInteractionEffectsCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target_column: str,
        features: list[str],
    ) -> dict[str, Any]:
        frame = _eager(data).select([target_column] + features).drop_nulls()
        y = frame[target_column].to_numpy()
        X = frame.select(features).to_numpy()
        
        from itertools import combinations
        
        inter_names = []
        inter_X = []
        
        # Base features
        for i, f in enumerate(features):
            inter_names.append(f)
            inter_X.append(X[:, i])
            
        # Interactions
        for f1, f2 in combinations(features, 2):
            idx1 = features.index(f1)
            idx2 = features.index(f2)
            inter_names.append(f"{f1}:{f2}")
            inter_X.append(X[:, idx1] * X[:, idx2])
            
        X_design = np.column_stack(inter_X)
        model = LinearRegression().fit(X_design, y)
        coefs = model.coef_
        
        effects = {}
        for name, coef in zip(inter_names, coefs):
            if ":" in name:
                effects[name] = float(coef)
                
        return {"interaction_effects": effects}


class MulticollinearityCalculatorPolars(AbstractMulticollinearityCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        columns: list[str],
    ) -> dict[str, Any]:
        frame = _eager(data).select(columns).drop_nulls()
        X = frame.to_numpy()
        
        # VIF = 1 / (1 - R^2)
        vifs = {}
        for i, col in enumerate(columns):
            y_i = X[:, i]
            X_i = np.delete(X, i, axis=1)
            
            model = LinearRegression().fit(X_i, y_i)
            r2 = model.score(X_i, y_i)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
            
            vifs[col] = float(vif)
            
        return {"vif_scores": vifs}


class MutualInformationCalculatorPolars(AbstractMutualInformationCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        target_column: str,
        feature_columns: list[str],
        is_target_discrete: bool = True,
    ) -> dict[str, Any]:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        frame = _eager(data).select([target_column] + feature_columns).drop_nulls()
        X = frame.select(feature_columns).to_numpy()
        y = frame[target_column].to_numpy()
        
        if is_target_discrete:
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
            
        res = {f: float(mi) for f, mi in zip(feature_columns, mi_scores)}
        return {"mutual_information": res}


class PartialCorrelationCalculatorPolars(AbstractPartialCorrelationCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        col1: str,
        col2: str,
        covariates: list[str],
    ) -> dict[str, Any]:
        import pingouin as pg
        
        frame = _eager(data).select([col1, col2] + covariates).drop_nulls()
        df_pd = frame.to_pandas()
        
        res = pg.partial_corr(data=df_pd, x=col1, y=col2, covar=covariates)
        
        r = float(res['r'].iloc[0])
        p = float(res['p-val'].iloc[0])
        
        return {
            "partial_correlation": r,
            "p_value": p,
            "x": col1,
            "y": col2,
            "covariates": covariates,
        }
