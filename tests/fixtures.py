"""Module for providing deterministic fixture data to unit tests"""

import numpy as np
import pandas as pd


class DataFrameFactory:
    """Fixture Factory to provide deterministic input datasets for testing."""

    @staticmethod
    def create_sentinels() -> pd.DataFrame:
        """Create a DataFrame containing various string sentinel values."""
        return pd.DataFrame({"mock_string": ["valid", "n/a", "nan", "valid2", "unknown"]})

    @staticmethod
    def create_dirty_bools() -> pd.DataFrame:
        """Create a DataFrame containing different boolean representations and NaNs."""
        return pd.DataFrame({"mock_bool": ["yes", "no", np.nan, "y"]})

    @staticmethod
    def create_outliers() -> pd.DataFrame:
        """Create a DataFrame with mixed id limits and target variable outliers."""
        return pd.DataFrame(
            {
                "mock_id": [1.0, 2.0, 50.0, 100.0, 9999.0],
                "mock_numeric": [1.0, 2.0, 50.0, 100.0, 9999.0],
            }
        )

    @staticmethod
    def create_dirty_text() -> pd.DataFrame:
        """Create a DataFrame with unstandardized text properties."""
        return pd.DataFrame({"mock_text": ["  TEXT  ", "Word_!", "café"]})

    @staticmethod
    def create_linear_regression_data() -> tuple[list[pd.Series], pd.DataFrame]:
        """
        Create a perfectly linear system dataframe dataset.
        Equation generated: y = 2 * x1 + 3 * x2 + 10
        """
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([5, 4, 3, 2, 1])
        y = 2 * x1 + 3 * x2 + 10

        x_series = [pd.Series(x1, name="x1"), pd.Series(x2, name="x2")]
        y_df = pd.DataFrame({"y": y})
        return x_series, y_df

    @staticmethod
    def create_categorical_data() -> pd.DataFrame:
        """Create a DataFrame containing standard categorical variables."""
        return pd.DataFrame({"mock_category": ["TypeA", "TypeB", "TypeA"]})
