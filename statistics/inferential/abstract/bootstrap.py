"""Abstract contract for non-parametric bootstrap estimation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable


class AbstractBootstrapEstimator(ABC):
    """Contract for non-parametric bootstrap estimation of a scalar statistic.

    Functionality
    -------------
    Uses random resampling with replacement to estimate the sampling
    distribution of a statistic, allowing for confidence interval construction
    without assuming a specific theoretical distribution.

    * **Observed statistic** — the statistic computed on the original sample.
    * **Confidence interval** — percentile-based lower and upper bounds.
    * **Bootstrap distribution** — mean, standard deviation (standard error),
      and bias of the resampled statistic distribution.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    column:
        Name of the numeric column to resample.
    statistic_expr:
        A callable or expression definition that tells the backend how to
        compute the target statistic on a column (e.g. mean, median).
        The exact type of this argument depends on the backend implementation.
    n_iterations:
        Number of bootstrap resamples to generate.
    confidence_level:
        Desired confidence level for the percentile interval (e.g. 0.95).
    random_seed:
        Seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Keys: ``observed_statistic``, ``confidence_interval`` (dict with
        ``lower``, ``upper``, ``confidence_level``), ``bootstrap_distribution``
        (dict with ``mean``, ``std``, ``bias``), ``n_iterations``, ``n``,
        ``random_seed``.

    Raises
    ------
    KeyError
        If ``column`` is not present in ``data``.
    ValueError
        If sample size is too small or iterations are insufficient.
    """

    @abstractmethod
    def estimate(
        self,
        data: Any,
        column: str,
        statistic_expr: Any,
        n_iterations: int = 5_000,
        confidence_level: float = 0.95,
        random_seed: int | None = 42,
    ) -> dict[str, Any]:
        """Run bootstrap estimation for ``statistic_expr`` on ``column``."""
