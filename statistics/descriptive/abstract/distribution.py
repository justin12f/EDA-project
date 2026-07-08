"""Abstract contract for distribution classification.

This module defines the abstract class that every backend must implement
to fit theoretical distributions against an empirical data column and
classify its shape.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractDistributionClassifier(ABC):
    """Contract for classifying the statistical distribution of a numeric column.

    Functionality
    -------------
    Orchestrates the following pipeline on a specified numeric column:

    1. **Distribution fitting** — fits candidate theoretical distributions
       (normal, log-normal, exponential, uniform, gamma, Weibull, Laplace,
       logistic) against the empirical data using a goodness-of-fit test
       (e.g. Kolmogorov-Smirnov statistic).  Distributions that require
       strictly positive data are skipped when the column contains
       non-positive values.

    2. **Bimodality detection** — applies the bimodality coefficient (BC)
       derived from skewness and excess kurtosis to flag two-peaked
       distributions.

    3. **Transformation advice** — recommends a data transformation
       (e.g. ``log1p``, ``sqrt``, ``box_cox``) based on the best-fit
       distribution name and the skewness magnitude.

    Implementations must use the native expression API of their backend and
    must not convert data to NumPy or an alien backend inside this method.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column to classify.

    Returns
    -------
    dict[str, Any]
        Keys:
        * ``best_fit`` — dict with ``name``, ``ks_statistic``, ``p_value``,
          ``parameters``.
        * ``all_fits`` — list of dicts with ``name``, ``ks_statistic``,
          ``p_value`` for every attempted distribution, sorted ascending by
          ``ks_statistic``.
        * ``is_bimodal`` — bool.
        * ``classification_label`` — str (``"bimodal"`` or best-fit name).
        * ``recommended_transformation`` — str or ``None``.
        * ``skewness`` — float.

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If the column has fewer than 8 non-null observations.
    RuntimeError
        If no distribution could be fitted.
    """

    @abstractmethod
    def classify(
        self,
        data: Any,
        column: str,
    ) -> dict[str, Any]:
        """Classify the distribution of ``column`` in ``data``."""
