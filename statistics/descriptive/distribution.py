# NOTE: vibecoded this script have multiple errors it doesn't validate
# the data before proccesing & the exeptions are stupid this thing use exept: Exception
# instead of specific exceptions but the math and logic is right
# TODO: read all the script and fix the errors 

"""Distribution classification and fitting module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DistributionFitResult:
    """Immutable result of a single distribution fit against the data."""

    name: str
    ks_statistic: float
    p_value: float
    parameters: dict[str, float]


class BimodalityDetector:
    """Detects bimodality using the bimodality coefficient (BC).

    The BC is derived from skewness and excess kurtosis. A value above
    the threshold (0.555) suggests bimodality.

    Reference:
        Pfister et al. (2013). Statistica Sinica.
    """

    _BIMODALITY_THRESHOLD: float = 0.555

    def detect(self, data: np.ndarray) -> bool:
        """Detect whether the distribution appears bimodal.

        Args:
            data: 1D numerical array with at least 4 elements.

        Returns:
            True if the bimodality coefficient exceeds the threshold.
        """
        n = len(data)
        skewness = float(stats.skew(data))
        excess_kurtosis = float(stats.kurtosis(data))

        numerator = skewness**2 + 1
        denominator = (
            excess_kurtosis + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
            if n > 3
            else excess_kurtosis + 3
        )

        bimodality_coefficient = numerator / denominator if denominator != 0 else 0.0
        return bimodality_coefficient > self._BIMODALITY_THRESHOLD


class TransformationAdvisor:
    """Recommends a data transformation based on distribution shape and skewness."""

    _DISTRIBUTION_RECOMMENDATIONS: dict[str, str] = {
        "log_normal": "log1p",
        "exponential": "log1p",
        "gamma": "sqrt or log1p",
        "weibull": "log1p",
    }

    def advise(self, best_fit_name: str, skewness: float) -> Optional[str]:
        """Return a recommended transformation string or None.

        Args:
            best_fit_name: Name of the best-fit distribution.
            skewness: Skewness value of the data.

        Returns:
            Transformation string or None if no transformation is needed.
        """
        if best_fit_name in self._DISTRIBUTION_RECOMMENDATIONS:
            return self._DISTRIBUTION_RECOMMENDATIONS[best_fit_name]

        if abs(skewness) > 1.0:
            return "box_cox or yeo_johnson"

        return None


class DistributionFitter:
    """Fits data against candidate theoretical distributions using the KS test.

    Distributions requiring strictly positive data are skipped if the data
    contains non-positive values.
    """

    _CANDIDATE_DISTRIBUTIONS: list[tuple[str, stats.rv_continuous]] = [
        ("normal", stats.norm),
        ("log_normal", stats.lognorm),
        ("exponential", stats.expon),
        ("uniform", stats.uniform),
        ("gamma", stats.gamma),
        ("weibull", stats.weibull_min),
        ("laplace", stats.laplace),
        ("logistic", stats.logistic),
    ]

    _POSITIVE_ONLY_DISTRIBUTIONS: frozenset[str] = frozenset(
        {"log_normal", "exponential", "gamma", "weibull"}
    )

    def fit_all(self, data: np.ndarray) -> list[DistributionFitResult]:
        """Fit all candidate distributions and return results sorted by KS statistic.

        Args:
            data: 1D numerical array.

        Returns:
            List of DistributionFitResult sorted ascending by ks_statistic.
        """
        results: list[DistributionFitResult] = []

        for name, distribution in self._CANDIDATE_DISTRIBUTIONS:
            if name in self._POSITIVE_ONLY_DISTRIBUTIONS and np.any(data <= 0):
                continue
            try:
                result = self._fit_single(name, distribution, data)
                if result is not None:
                    results.append(result)
            except Exception:
                continue

        return sorted(results, key=lambda r: r.ks_statistic)

    def _fit_single(
        self,
        name: str,
        distribution: stats.rv_continuous,
        data: np.ndarray,
    ) -> Optional[DistributionFitResult]:
        """Fit a single distribution to the data.

        Args:
            name: Human-readable distribution name.
            distribution: SciPy continuous distribution object.
            data: 1D numerical array.

        Returns:
            DistributionFitResult or None if fitting fails.
        """
        params = distribution.fit(data)
        ks_statistic, p_value = stats.kstest(data, distribution.cdf, args=params)

        shape_params: list[str] = (
            distribution.shapes.split(", ") if distribution.shapes else []
        )
        all_param_names = shape_params + ["loc", "scale"]
        parameters = {k: float(v) for k, v in zip(all_param_names, params)}

        return DistributionFitResult(
            name=name,
            ks_statistic=float(ks_statistic),
            p_value=float(p_value),
            parameters=parameters,
        )


class DistributionClassifier:
    """Orchestrates the full distribution classification pipeline.

    Workflow:
        classifier = DistributionClassifier()
        result = classifier.classify(data, significance_level=0.05)

    Returns a dict with keys:
        - best_fit: dict with name, ks_statistic, p_value, parameters
        - all_fits: list of dicts with name, ks_statistic, p_value
        - is_bimodal: bool
        - classification_label: str
        - recommended_transformation: str or None
        - skewness: float
    """

    _MINIMUM_SAMPLE_SIZE: int = 8

    def __init__(self) -> None:
        self._fitter = DistributionFitter()
        self._bimodality_detector = BimodalityDetector()
        self._transformation_advisor = TransformationAdvisor()

    def classify(self, data: np.ndarray) -> dict:
        """Classify the statistical distribution of a 1D array.

        Args:
            data: Clean 1D numerical array (no NaN).

        Returns:
            Dictionary with classification results.

        Raises:
            ValueError: If data has fewer than the minimum required samples.
            RuntimeError: If no distribution could be fitted.
        """
        if len(data) < self._MINIMUM_SAMPLE_SIZE:
            raise ValueError(
                f"Distribution classification requires at least "
                f"{self._MINIMUM_SAMPLE_SIZE} samples. Got {len(data)}."
            )

        all_fits = self._fitter.fit_all(data)

        if not all_fits:
            raise RuntimeError("No distributions could be fitted to the provided data.")

        best_fit = all_fits[0]
        is_bimodal = self._bimodality_detector.detect(data)
        skewness = float(stats.skew(data))
        label = "bimodal" if is_bimodal else best_fit.name
        transformation = self._transformation_advisor.advise(best_fit.name, skewness)

        return {
            "best_fit": {
                "name": best_fit.name,
                "ks_statistic": best_fit.ks_statistic,
                "p_value": best_fit.p_value,
                "parameters": best_fit.parameters,
            },
            "all_fits": [
                {
                    "name": r.name,
                    "ks_statistic": r.ks_statistic,
                    "p_value": r.p_value,
                }
                for r in all_fits
            ],
            "is_bimodal": is_bimodal,
            "classification_label": label,
            "recommended_transformation": transformation,
            "skewness": skewness,
        }
