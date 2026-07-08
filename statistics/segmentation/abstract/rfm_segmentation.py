"""Abstract contract for RFM Segmentation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AbstractRFMSegmentationCalculator(ABC):
    """Contract for Recency, Frequency, Monetary (RFM) Segmentation.

    Parameters
    ----------
    data:
        Backend-native dataframe with transactional data.
    customer_column:
        Customer identifier.
    date_column:
        Transaction date.
    amount_column:
        Transaction amount/value.
    reference_date:
        Reference date for Recency (usually max date in dataset).

    Returns
    -------
    dict[str, Any]
        Dictionary with RFM scores, segments, and summary statistics.
    """

    @abstractmethod
    def calculate(
        self,
        data: Any,
        customer_column: str,
        date_column: str,
        amount_column: str,
        reference_date: str | None = None,
    ) -> dict[str, Any]:
        """Perform RFM segmentation."""
