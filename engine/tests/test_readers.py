"""Reader factory tests."""

import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumen.readers.reader_factory import ReaderFactory


class TestReaders(unittest.TestCase):
    def test_pandas_csv_roundtrip(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,2\n")
            path = f.name
        try:
            reader = ReaderFactory.create(path, backend="pandas")
            df = reader.read()
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(list(df.columns), ["a", "b"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
