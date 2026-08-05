"""Module for testing data cleaning capabilities relying on pipeline interfaces."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumen.data_cleaning.data_cleaning_step_factory import DataCleaningStepFactory
from tests.fixtures import DataFrameFactory


class BaseStepTest(unittest.TestCase):
    """
    SRP: Base Engine for injecting data frames into the abstract Step pipelines.
    Configures common factory retrieval and execution mechanisms contexts.
    """

    def execute_pipeline_step(self, step_name: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Dependency Injection pattern used through the DataCleaningStepFactory."""
        step = DataCleaningStepFactory.create(step_name, df, **kwargs)
        return step.process(df)


class TestDataCleaningPipeline(BaseStepTest):
    """SRP: Exclusively dedicated to verifying Data Cleaning Transformations."""

    def test_handle_sentinels(self):
        """Verifies mathematical replacement of textual placeholder NaNs strings."""
        df = DataFrameFactory.create_sentinels()
        cleaned = self.execute_pipeline_step("handle_sentinel_values", df)

        self.assertTrue(pd.isna(cleaned["mock_string"].iloc[1]))
        self.assertTrue(pd.isna(cleaned["mock_string"].iloc[2]))
        self.assertTrue(pd.isna(cleaned["mock_string"].iloc[4]))
        self.assertEqual(cleaned["mock_string"].iloc[0], "valid")

    def test_fix_bools_preserves_nans(self):
        """Validates correct boolean evaluation mapping retaining mathematical nulls accurately."""
        df = DataFrameFactory.create_dirty_bools()
        cleaned = self.execute_pipeline_step("fix_bools_columns", df)

        self.assertTrue(cleaned["mock_bool"].iloc[0])
        self.assertFalse(cleaned["mock_bool"].iloc[1])
        self.assertTrue(pd.isna(cleaned["mock_bool"].iloc[2]))
        self.assertEqual(cleaned["mock_bool"].dtype, "boolean")

    def test_cap_outliers_ignores_ids(self):
        """Securing Winsorization bounds correctly avoid computing variables tagged as IDs."""
        df = DataFrameFactory.create_outliers()
        cleaned = self.execute_pipeline_step("cap_outliers", df, upper_percentile=0.90)

        self.assertEqual(cleaned["mock_id"].iloc[-1], 9999.0)
        self.assertLess(cleaned["mock_numeric"].iloc[-1], 9999.0)

    def test_text_standardization(self):
        """Verifies uniform text sanitation regarding extra whitespaces and unallowed strings."""
        df = DataFrameFactory.create_dirty_text()
        cleaned = self.execute_pipeline_step("text_standardization", df)

        self.assertEqual(cleaned["mock_text"].iloc[0], "text")
        self.assertEqual(cleaned["mock_text"].iloc[1], "word_")
        self.assertEqual(cleaned["mock_text"].iloc[2], "cafe")


if __name__ == "__main__":
    unittest.main()
