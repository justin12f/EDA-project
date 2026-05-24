"""Analyzer factory smoke tests."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyze_data.analyzer_factory import DataAnalyzerFactory


class TestAnalyzers(unittest.TestCase):
    def test_shape_pandas(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        analyzer = DataAnalyzerFactory.create_analyzer("shape", "pandas", df)
        result = analyzer.analyze()
        self.assertIn("rows", str(result).lower() + str(result))


if __name__ == "__main__":
    unittest.main()
