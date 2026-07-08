"""Abstract contract for Conversion Funnel calculation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractConversionFunnelCalculator(ABC):
    """Contract for calculating conversion funnel metrics.

    Parameters
    ----------
    data:
        Backend-native dataframe.
    step_column:
        Name of the column containing funnel step names.
    user_column:
        Name of the column containing user identifiers.
    steps_order:
        List of step names in the correct funnel order.

    Returns
    -------
    dict[str, Any]
        Dictionary with step-by-step conversion rates and overall conversion.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        step_column: str,
        user_column: str,
        steps_order: list[str],
    ) -> dict[str, Any]:
        """Calculate funnel conversion metrics."""
