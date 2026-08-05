"""CLI argument parser for data cleaning pipelines."""

from __future__ import annotations

import argparse

from lumen.core.backend import BACKENDS, DEFAULT_BACKEND


def build_parser(file_path: str, preset: str = "default") -> argparse.ArgumentParser:
    """Build argument parser (does not parse argv)."""
    parser_object = argparse.ArgumentParser(description="Run data cleaning pipeline")
    parser_object.add_argument(
        "-i", "--input", type=str, default=file_path, help="Input file path"
    )
    parser_object.add_argument(
        "-o", "--output", type=str, help="Output file path (default: clean_<input>)"
    )
    parser_object.add_argument(
        "-p",
        "--preset",
        type=str,
        choices=["light", "default", "strict"],
        default=preset,
        help="Pipeline preset configuration",
    )
    parser_object.add_argument(
        "-r",
        "--report",
        type=str,
        help="Path to save the JSON report (default: cleaning_report_<input>.json)",
    )
    parser_object.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=list(BACKENDS),
        default=DEFAULT_BACKEND,
        help="Compute backend for readers and cleaning pipeline",
    )
    return parser_object


def parser(file_path: str, preset: str = "default") -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser(file_path, preset).parse_args()
