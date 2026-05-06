"""Module for abstract classes"""

import pandas as pd

from .base import BaseEncoder, BaseTransform


class GetColumns:
    """Get the columns of the data frame"""

    def get_columns(self, data: pd.DataFrame) -> list[str]:
        """Fit the encoder to the data"""
        string_columns = data.select_dtypes(include=["object", "string"]).columns.tolist()
        return string_columns


class GetCategories:
    """Get the categories of the data frame"""

    def get_categories(
        self, data: pd.DataFrame, list_of_columns: list[str]
    ) -> dict[str, list[str]]:
        """Fit the encoder to the data"""
        columns_categories: dict[str, list[str]]
        columns_categories = {column: data[column].unique() for column in list_of_columns}
        return columns_categories


class OneHotEncoderTransform(BaseTransform):
    """Transforming logic for OneHotEncoder"""

    def __init__(self, data_frame: pd.DataFrame, columns_categories: dict[str, list[str]]) -> None:
        self.data_frame = data_frame.copy()
        self.columns_categories = columns_categories

    def transform(self) -> pd.DataFrame:
        """transform the unique values into columns"""
        data_frame_output = self.data_frame.copy()

        for column, categories in self.columns_categories.items():
            for category in categories:
                data_frame_output[f"{column}_{category}"] = (
                    data_frame_output[column] == category
                ).astype(int)

        columns_to_drop = list(self.columns_categories.keys())
        data_frame_output = data_frame_output.drop(columns=columns_to_drop)

        return data_frame_output


class OneHotEncoder(BaseEncoder):
    """One hot encoder make all the diferent values of a column into columns
    with boolean values 1 if  the row have that  & value and 0 if it is not"""

    def __init__(self) -> None:
        self.columns_categories: dict[str, list[str]] | None = None
        self.data_frame: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the encoder to the data"""
        columns = GetColumns().get_columns(data)
        self.columns_categories = GetCategories().get_categories(data, columns)
        self.data_frame = data

    def transform(self) -> pd.DataFrame:
        if self.columns_categories is None or self.data_frame is None:
            raise ValueError("The encoder must be Fitted before transformation")

        return_value = OneHotEncoderTransform(self.data_frame, self.columns_categories).transform()

        return return_value


class OrdinalEncoderTransform(BaseTransform):
    """Transforming logic for OrdinalEncoder"""

    def __init__(
        self, data_frame: pd.DataFrame, columns_categories_hierarchy: dict[str, list[str]]
    ) -> None:
        self.data_frame = data_frame.copy()
        self.mappings = {
            col: {val: i for i, val in enumerate(hierarchy)}
            for col, hierarchy in columns_categories_hierarchy.items()
        }

    def transform(self) -> pd.Series:
        """transform the unique vaalues into columns"""
        data_frame_output = self.data_frame.copy()
        columns = GetColumns().get_columns(self.data_frame)

        for column, mapping in self.mappings.items():
            if column in columns:
                data_frame_output[column] = data_frame_output[column].map(mapping)

            if data_frame_output[column].isnull().any():
                raise ValueError(
                    f"The value {data_frame_output[column].unique()} is not in the hierarchy"
                )

        return data_frame_output


class OrdinalEncoder(BaseEncoder):
    """Ordinal encoder encode the values of a column into integers"""

    def __init__(self) -> None:
        self.columns_hierarchy: dict[str, list[str]] | None = None
        self.data_frame: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the encoder to the data"""
        columns_hierarchy = kwargs.get("columns_categories_hierarchy", {})
        self.data_frame = data
        self.columns_hierarchy = columns_hierarchy

    def transform(self) -> pd.Series:
        if self.columns_hierarchy is None:
            raise ValueError("The encoder must be Fitted before transformation")

        return_value = OrdinalEncoderTransform(self.data_frame, self.columns_hierarchy).transform()
        return return_value
