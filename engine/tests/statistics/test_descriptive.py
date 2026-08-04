"""Descriptive statistics factory tests."""

import os
import sys
import unittest

import pandas as pd
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lumen.statistics.descriptive.factory import DescriptiveStatisticsFactory
from lumen.statistics.inyeccion import StatisticsInyeccionDependency


class TestDescriptiveStatistics(unittest.TestCase):
    def test_mean_pandas(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        calc = DescriptiveStatisticsFactory.create("mean_calculator", "pandas")
        self.assertAlmostEqual(calc.calculate(df, "x"), 2.0)

    def test_mean_polars_native(self) -> None:
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        calc = DescriptiveStatisticsFactory.create("mean_calculator", "polars")
        self.assertAlmostEqual(calc.calculate(df, "x"), 2.0)

    def test_inyeccion_distribution_classify(self) -> None:
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        result = StatisticsInyeccionDependency("polars").run(
            "descriptive",
            "distribution_classifier",
            df,
            column="x",
            method="classify",
        )
        self.assertIn("classification_label", result)


if __name__ == "__main__":
    unittest.main()
