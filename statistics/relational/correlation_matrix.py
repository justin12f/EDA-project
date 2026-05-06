"""Full correlation matrix with ranking and filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CorrelationPair:
    """Immutable representation of a pairwise correlation result."""

    column_a: str
    column_b: str
    coefficient: float
    abs_coefficient: float
    method: str


class CorrelationMatrixComputer:
    """Computes a full correlation matrix using the specified method.

    Supported methods:
        - 'pearson':  Linear correlation. Assumes bivariate normality.
        - 'spearman': Rank-based. Robust to outliers and monotone relationships.
        - 'kendall':  Rank concordance. More robust for small samples.
    """

    _VALID_METHODS: frozenset[str] = frozenset({"pearson", "spearman", "kendall"})

    def compute(
        self,
        data_frame: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        """Compute the correlation matrix.

        Args:
            data_frame: DataFrame with only numeric columns.
            method: Correlation method.

        Returns:
            Symmetric correlation matrix as a DataFrame.

        Raises:
            ValueError: If method is not supported or fewer than 2 columns.
        """
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}. Got '{method}'."
            )
        if data_frame.shape[1] < 2:
            raise ValueError(
                "At least 2 numeric columns are required to compute a correlation matrix."
            )

        return data_frame.corr(method=method)


class CorrelationPairExtractor:
    """Extracts and ranks unique pairwise correlations from a matrix."""

    def extract(
        self,
        matrix: pd.DataFrame,
        method: str,
        top_n: int | None,
        threshold: float,
    ) -> list[dict]:
        """Extract sorted pairwise correlations from the upper triangle.

        Args:
            matrix: Square symmetric correlation matrix.
            method: Method name for labeling.
            top_n: Limit results to top N pairs by |coefficient|.
            threshold: Only include pairs with |coefficient| >= threshold.

        Returns:
            List of correlation pair dicts sorted descending by |coefficient|.
        """
        pairs: list[CorrelationPair] = []

        columns = matrix.columns.tolist()
        for i, col_a in enumerate(columns):
            for col_b in columns[i + 1:]:
                coefficient = float(matrix.loc[col_a, col_b])
                if np.isnan(coefficient):
                    continue
                abs_coeff = abs(coefficient)
                if abs_coeff < threshold:
                    continue
                pairs.append(
                    CorrelationPair(
                        column_a=col_a,
                        column_b=col_b,
                        coefficient=coefficient,
                        abs_coefficient=abs_coeff,
                        method=method,
                    )
                )

        pairs.sort(key=lambda p: p.abs_coefficient, reverse=True)

        if top_n is not None:
            pairs = pairs[:top_n]

        return [
            {
                "column_a": p.column_a,
                "column_b": p.column_b,
                "coefficient": p.coefficient,
                "abs_coefficient": p.abs_coefficient,
                "method": p.method,
            }
            for p in pairs
        ]


class HighCorrelationFlagDetector:
    """Flags column pairs that exceed a multicollinearity risk threshold."""

    _DEFAULT_HIGH_CORRELATION_THRESHOLD: float = 0.85

    def detect(
        self,
        pairs: list[dict],
        high_threshold: float,
    ) -> list[dict]:
        """Filter pairs that exceed the high-correlation threshold.

        Args:
            pairs: Already-extracted pair list.
            high_threshold: Absolute coefficient threshold for flagging.

        Returns:
            Subset of pairs with |coefficient| >= high_threshold.
        """
        return [p for p in pairs if p["abs_coefficient"] >= high_threshold]


class CorrelationMatrixCalculator:
    """Full correlation matrix analysis with ranking and risk flagging.

    Workflow:
        calculator = CorrelationMatrixCalculator()
        result = calculator.calculate(
            data_frame=df[["col_a", "col_b", "col_c"]],
            method="pearson",          # "pearson" | "spearman" | "kendall"
            top_n=10,                  # optional
            threshold=0.0,             # minimum |r| to include in pairs list
            high_correlation_flag=0.85 # threshold for collinearity warning
        )
    """

    def __init__(self) -> None:
        self._matrix_computer = CorrelationMatrixComputer()
        self._pair_extractor = CorrelationPairExtractor()
        self._flag_detector = HighCorrelationFlagDetector()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        method: str = "pearson",
        top_n: int | None = None,
        threshold: float = 0.0,
        high_correlation_flag: float = 0.85,
    ) -> dict:
        """Run full correlation matrix analysis.

        Args:
            data_frame: Numeric-only DataFrame.
            method: Correlation method.
            top_n: Return only top N pairs. None = all pairs.
            threshold: Minimum |coefficient| to include in output pairs.
            high_correlation_flag: Threshold for multicollinearity warning.

        Returns:
            Dict with matrix, ranked pairs, flagged pairs, and metadata.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0.0, 1.0]. Got {threshold}."
            )
        if not 0.0 <= high_correlation_flag <= 1.0:
            raise ValueError(
                f"high_correlation_flag must be in [0.0, 1.0]. "
                f"Got {high_correlation_flag}."
            )

        matrix = self._matrix_computer.compute(data_frame, method)
        pairs = self._pair_extractor.extract(matrix, method, top_n, threshold)
        flagged = self._flag_detector.detect(pairs, high_correlation_flag)

        return {
            "matrix": matrix.to_dict(),
            "ranked_pairs": pairs,
            "high_correlation_pairs": flagged,
            "n_pairs_total": len(pairs),
            "n_flagged": len(flagged),
            "method": method,
            "threshold": threshold,
            "high_correlation_flag": high_correlation_flag,
            "columns_analysed": data_frame.columns.tolist(),
        }
