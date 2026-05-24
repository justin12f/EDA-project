"""Tests for AgentMasterFactory and backend validation."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.master_factory import AgentMasterFactory
from core.backend import validate_backend


class TestAgentMasterFactory(unittest.TestCase):
    def test_validate_backend_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            validate_backend("duckdb")

    def test_master_exposes_same_backend(self) -> None:
        master = AgentMasterFactory("pandas")
        self.assertEqual(master.backend, "pandas")
        self.assertEqual(master.readers().backend, "pandas")

    def test_readers_factory_lists_pandas(self) -> None:
        from readers.reader_factory import ReaderFactory

        self.assertIn("pandas", ReaderFactory.available_backends())


if __name__ == "__main__":
    unittest.main()
