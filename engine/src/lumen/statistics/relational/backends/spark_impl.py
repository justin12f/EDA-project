"""PySpark-native backend implementations for the relational statistics domain."""
from __future__ import annotations

from typing import Any
import math
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from lumen.statistics.relational.abstract.contingency_analysis import AbstractContingencyAnalysisCalculator
from lumen.statistics.relational.abstract.correlation_matrix import AbstractCorrelationMatrixCalculator
from lumen.statistics.relational.abstract.cross_correlation import AbstractCrossCorrelationCalculator
from lumen.statistics.relational.abstract.granger_causality import AbstractGrangerCausalityCalculator
from lumen.statistics.relational.abstract.interaction_effects import AbstractInteractionEffectsCalculator
from lumen.statistics.relational.abstract.multicollinearity import AbstractMulticollinearityCalculator
from lumen.statistics.relational.abstract.mutual_information import AbstractMutualInformationCalculator
from lumen.statistics.relational.abstract.partial_correlation import AbstractPartialCorrelationCalculator


class ContingencyAnalysisCalculatorSpark(AbstractContingencyAnalysisCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        col1: str,
        col2: str,
    ) -> dict[str, Any]:
        crosstab_df = data.dropna(subset=[col1, col2]).crosstab(col1, col2)
        rows = crosstab_df.collect()
        
        matrix = np.array([ [r[c] for c in crosstab_df.columns[1:]] for r in rows ], dtype=float)
        
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


class CorrelationMatrixCalculatorSpark(AbstractCorrelationMatrixCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        columns: list[str],
        method: str = "pearson",
    ) -> dict[str, Any]:
        from pyspark.ml.stat import Correlation
        from pyspark.ml.feature import VectorAssembler
        
        clean = data.select(columns).dropna()
        n = clean.count()
        
        assembler = VectorAssembler(inputCols=columns, outputCol="features")
        df_vector = assembler.transform(clean).select("features")
        
        corr_row = Correlation.corr(df_vector, "features", method=method).head()
        matrix = corr_row[0].toArray()
        
        corr_matrix = []
        p_matrix = []
        
        for i in range(len(columns)):
            c_row = []
            p_row = []
            for j in range(len(columns)):
                corr = float(matrix[i, j])
                c_row.append(corr)
                
                if i == j:
                    p_row.append(0.0)
                else:
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


class CrossCorrelationCalculatorSpark(AbstractCrossCorrelationCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        col1: str,
        col2: str,
        max_lag: int = 10,
    ) -> dict[str, Any]:
        # Spark lacks inherent ordering without a column, assuming we can collect for time series analysis
        # For large data, we'd need a time column for Window.orderBy
        # Since it's a general time series function, collecting is common
        clean = data.select(col1, col2).dropna()
        rows = clean.collect()
        
        arr1 = np.array([r[0] for r in rows], dtype=float)
        arr2 = np.array([r[1] for r in rows], dtype=float)
        
        n = len(arr1)
        res = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                c1 = arr1[:lag]
                c2 = arr2[-lag:]
            elif lag > 0:
                c1 = arr1[lag:]
                c2 = arr2[:-lag]
            else:
                c1 = arr1
                c2 = arr2
                
            corr = float(np.corrcoef(c1, c2)[0, 1]) if len(c1) > 1 else 0.0
            res.append({"lag": lag, "correlation": corr})

        return {"cross_correlations": res, "max_lag": max_lag}


class GrangerCausalityCalculatorSpark(AbstractGrangerCausalityCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        target_column: str,
        predictor_column: str,
        max_lag: int = 5,
    ) -> dict[str, Any]:
        from statsmodels.tsa.stattools import grangercausalitytests
        
        clean = data.select(target_column, predictor_column).dropna()
        rows = clean.collect()
        arr = np.array([[r[0], r[1]] for r in rows], dtype=float)
        
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


class InteractionEffectsCalculatorSpark(AbstractInteractionEffectsCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        target_column: str,
        features: list[str],
    ) -> dict[str, Any]:
        from itertools import combinations
        
        clean = data.select([target_column] + features).dropna()
        
        # Adding interaction columns via expressions
        df = clean
        inter_names = list(features)
        for f1, f2 in combinations(features, 2):
            col_name = f"{f1}:{f2}"
            df = df.withColumn(col_name, F.col(f1) * F.col(f2))
            inter_names.append(col_name)
            
        from pyspark.ml.regression import LinearRegression as SparkLR
        from pyspark.ml.feature import VectorAssembler
        
        assembler = VectorAssembler(inputCols=inter_names, outputCol="features")
        df_vector = assembler.transform(df).select(F.col(target_column).alias("label"), "features")
        
        lr = SparkLR(featuresCol="features", labelCol="label", solver="normal")
        model = lr.fit(df_vector)
        
        coefs = model.coefficients.toArray()
        
        effects = {}
        for name, coef in zip(inter_names, coefs):
            if ":" in name:
                effects[name] = float(coef)
                
        return {"interaction_effects": effects}


class MulticollinearityCalculatorSpark(AbstractMulticollinearityCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        columns: list[str],
    ) -> dict[str, Any]:
        from pyspark.ml.regression import LinearRegression as SparkLR
        from pyspark.ml.feature import VectorAssembler
        
        clean = data.select(columns).dropna()
        vifs = {}
        
        for i, col in enumerate(columns):
            predictors = [c for c in columns if c != col]
            assembler = VectorAssembler(inputCols=predictors, outputCol="features")
            df_vector = assembler.transform(clean).select(F.col(col).alias("label"), "features")
            
            lr = SparkLR(featuresCol="features", labelCol="label", solver="normal", maxIter=10)
            try:
                model = lr.fit(df_vector)
                r2 = model.summary.r2
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
            except Exception:
                vif = float('inf')
                
            vifs[col] = float(vif)
            
        return {"vif_scores": vifs}


class MutualInformationCalculatorSpark(AbstractMutualInformationCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        target_column: str,
        feature_columns: list[str],
        is_target_discrete: bool = True,
    ) -> dict[str, Any]:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        # Scikit-learn is required for MI estimation using nearest neighbors, 
        # which isn't available natively in PySpark MLlib.
        clean = data.select([target_column] + feature_columns).dropna()
        rows = clean.collect()
        
        y = np.array([r[0] for r in rows], dtype=float)
        X = np.array([[r[i+1] for i in range(len(feature_columns))] for r in rows], dtype=float)
        
        if is_target_discrete:
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
            
        res = {f: float(mi) for f, mi in zip(feature_columns, mi_scores)}
        return {"mutual_information": res}


class PartialCorrelationCalculatorSpark(AbstractPartialCorrelationCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        col1: str,
        col2: str,
        covariates: list[str],
    ) -> dict[str, Any]:
        import pingouin as pg
        
        clean = data.select([col1, col2] + covariates).dropna()
        df_pd = clean.toPandas()
        
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
