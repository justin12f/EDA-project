"""Pandas encoder implementations."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SklearnOHE
from sklearn.preprocessing import OrdinalEncoder as SklearnOE

from lumen.preproccesing.encoders.implementations.base import AbstractEncoder


class PandasOneHotEncoder(AbstractEncoder[pd.DataFrame]):
    def __init__(self) -> None:
        self._encoder = SklearnOHE(sparse_output=False, handle_unknown="ignore")
        self._columns: list[str] = []
        self._data: pd.DataFrame | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> None:
        columns = kwargs.get("columns")
        self._columns = columns or list(
            data.select_dtypes(include=["object", "string"]).columns
        )
        self._data = data
        if self._columns:
            self._encoder.fit(data[self._columns])
        self._fitted = True

    def transform(self) -> pd.DataFrame:
        if not self._fitted or self._data is None:
            raise RuntimeError("Encoder must be fitted before transform.")
        if not self._columns:
            return pd.DataFrame(index=self._data.index)
        encoded = self._encoder.transform(self._data[self._columns])
        return pd.DataFrame(
            encoded,
            columns=self._encoder.get_feature_names_out(self._columns),
            index=self._data.index,
        )


class PandasOrdinalEncoder(AbstractEncoder[pd.DataFrame]):
    def __init__(self) -> None:
        self._encoder = SklearnOE(handle_unknown="use_encoded_value", unknown_value=-1)
        self._columns: list[str] = []
        self._data: pd.DataFrame | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> None:
        columns = kwargs.get("columns")
        self._columns = columns or list(
            data.select_dtypes(include=["object", "string"]).columns
        )
        self._data = data
        if self._columns:
            self._encoder.fit(data[self._columns])
        self._fitted = True

    def transform(self) -> pd.DataFrame:
        if not self._fitted or self._data is None:
            raise RuntimeError("Encoder must be fitted before transform.")
        if not self._columns:
            return pd.DataFrame(index=self._data.index)
        encoded = self._encoder.transform(self._data[self._columns])
        return pd.DataFrame(encoded, columns=self._columns, index=self._data.index)
