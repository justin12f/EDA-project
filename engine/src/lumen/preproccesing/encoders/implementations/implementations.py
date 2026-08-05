"""Encoder implementations — backwards-compatible re-exports.

Re-exports Polars implementations as default for legacy import paths.
"""

from lumen.preproccesing.encoders.polars_impl import (
    PolarsOneHotEncoder as OneHotEncoder,
    PolarsOrdinalEncoder as OrdinalEncoder,
    PolarsGetColumns as GetColumns,
    PolarsGetCategories as GetCategories,
)

__all__ = ["OneHotEncoder", "OrdinalEncoder", "GetColumns", "GetCategories"]
