"""Standard scaler — backwards-compatible import alias.

This module re-exports the Polars implementation as the default
StandardScaler for callers that import from this legacy path.
"""

from lumen.preproccesing.scalers.polars_impl import PolarsStandardScaler as StandardScaler

__all__ = ["StandardScaler"]
