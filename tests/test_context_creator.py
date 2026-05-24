"""Smoke tests for context creator agent wiring."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.context_creator import ContextCreatorAgent, tools_openai


class TestContextCreator(unittest.TestCase):
    def test_tools_registered(self) -> None:
        self.assertEqual(len(tools_openai), 3)

    def test_agent_init(self) -> None:
        agent = ContextCreatorAgent("/tmp/sample.csv")
        self.assertEqual(agent.file_path, "/tmp/sample.csv")
        self.assertEqual(len(agent.tools), 3)


if __name__ == "__main__":
    unittest.main()
