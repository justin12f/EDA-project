"""Module for testing linear regression logic and evaluation architectures."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumen.models.linear_regression import BuildDesignMatrix, LinearRegressionFactory
from tests.fixtures import DataFrameFactory


class TestLinearRegressionModels(unittest.TestCase):
    """SRP: Test suite exclusively dedicated to Linear Regression algorithmic evaluation."""

    def test_build_design_matrix_structural_resolution(self):
        """Verifies proper unified matrix resolution logic processing array configurations."""
        x = DataFrameFactory.create_linear_regression_data()[0]
        matrix = BuildDesignMatrix().build_design_matrix(x)

        # We ensure a unified numpy 2D array output: A column of 1.0 (intercept factor) and exactly two predictive columns
        self.assertEqual(matrix.shape, (5, 3))
        self.assertEqual(matrix[0, 0], 1.0)
        self.assertEqual(matrix[0, 1], 1.0)  # Matches first value of X1 array
        self.assertEqual(matrix[0, 2], 5.0)  # Matches first value of X2 array

    def test_ordinary_least_squares_performance(self):
        """Testing analytical performance and predicting mathematical exactness natively."""
        x, y = DataFrameFactory.create_linear_regression_data()

        # Dependency Injection & Factory resolution logic
        model = LinearRegressionFactory.create("ordinary_least_squares", "multiple")
        model.fit(x, y)

        predictions = model.predict(x)
        scores = model.score(y)

        self.assertTrue(len(predictions) == len(y))

        # Validating performance: R^2 must be perfect (1.0) as the data equation correlates 1:1 perfectly
        self.assertAlmostEqual(scores["squared_r"], 1.0, places=3)
        # Validating loss performance calculation
        self.assertAlmostEqual(scores["mean_square_error"], 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
