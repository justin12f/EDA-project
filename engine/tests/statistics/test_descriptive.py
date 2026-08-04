"""Descriptive statistics factory tests.

`mean_calculator` no longer exists as a registry key: the per-statistic classes
were consolidated into one `central_tendency_calculator` per backend, which
returns mean, median, mode and trimmed mean together. These tests assert the
current contract — the same numbers, through the key the factory registers.
"""

import unittest

import pandas as pd
import polars as pl

from lumen.statistics.descriptive.factory import DescriptiveStatisticsFactory
from lumen.statistics.inyeccion import StatisticsInyeccionDependency
from lumen.statistics.registry import StatisticsRegistry, available_domains


class TestDescriptiveStatistics(unittest.TestCase):
    def test_central_tendency_pandas(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        calc = DescriptiveStatisticsFactory.create("central_tendency_calculator", "pandas")
        result = calc.calculate(df, "x")
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["median"], 2.0)

    def test_central_tendency_polars_native(self) -> None:
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        calc = DescriptiveStatisticsFactory.create("central_tendency_calculator", "polars")
        result = calc.calculate(df, "x")
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["median"], 2.0)

    def test_both_backends_agree(self) -> None:
        values = [1.0, 2.0, 3.0, 10.0]
        pandas_result = DescriptiveStatisticsFactory.create(
            "central_tendency_calculator", "pandas"
        ).calculate(pd.DataFrame({"x": values}), "x")
        polars_result = DescriptiveStatisticsFactory.create(
            "central_tendency_calculator", "polars"
        ).calculate(pl.DataFrame({"x": values}), "x")
        self.assertAlmostEqual(pandas_result["mean"], polars_result["mean"])
        self.assertAlmostEqual(pandas_result["median"], polars_result["median"])

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

    def test_an_unknown_key_names_what_was_asked_for(self) -> None:
        with self.assertRaises(ValueError) as caught:
            DescriptiveStatisticsFactory.create("mean_calculator", "polars")
        self.assertIn("mean_calculator", str(caught.exception))


class TestRegistryDiscovery(unittest.TestCase):
    """A domain that cannot load must not take the others down with it."""

    def test_descriptive_is_discoverable_through_the_unified_registry(self) -> None:
        self.assertIn("descriptive", available_domains())
        self.assertTrue(
            StatisticsRegistry.is_registered("descriptive.central_tendency_calculator", "polars")
        )

    def test_pandas_and_polars_are_registered_without_pyspark(self) -> None:
        backends = {backend for _, backend in StatisticsRegistry._registry}
        self.assertIn("pandas", backends)
        self.assertIn("polars", backends)


if __name__ == "__main__":
    unittest.main()
