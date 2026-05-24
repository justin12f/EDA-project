"""Module exclusively dedicated to verifying ML encoding strategies using Factory Injection."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preproccesing.encoders.encoder_factory import EncoderFactory
from tests.fixtures import DataFrameFactory


class TestEncoders(unittest.TestCase):
    """SRP: Test suite exclusively dedicated to Encoding and String Transformers."""

    def test_one_hot_encoder_mapping(self):
        """Testing correct symmetric output matrix column-dimensions over dummy categories."""
        df = DataFrameFactory.create_categorical_data()

        # We inject the data and encoder type using the existing architectural Factory
        encoder = EncoderFactory.create("one_hot", backend="pandas")
        encoder.fit(df)
        encoded_df = encoder.transform()

        self.assertIn("mock_category_TypeA", encoded_df.columns)
        self.assertIn("mock_category_TypeB", encoded_df.columns)
        self.assertEqual(encoded_df["mock_category_TypeA"].iloc[0], 1)
        self.assertEqual(encoded_df["mock_category_TypeB"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
