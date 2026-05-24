#!/usr/bin/env python3
"""Patch polars_impl.py: native polars inspectors + statistics registry."""

from __future__ import annotations

import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "analyze_data/analyzers/backends/polars_impl.py"

CALC_MAP = {
    "DistributionClassifier": ("descriptive", "distribution_classifier", "classify"),
    "NormalityTestSuite": ("descriptive", "normality_test_suite", "run"),
    "CentralTendencyCalculator": ("descriptive", "central_tendency_calculator", "calculate"),
    "DispersionCalculator": ("descriptive", "dispersion_calculator", "calculate"),
    "FrequencyDistributionBuilder": ("descriptive", "frequency_distribution_builder", "build"),
    "PercentilesCalculator": ("descriptive", "percentiles_calculator", "calculate"),
    "SkewnessKurtosisCalculator": ("descriptive", "skewness_kurtosis_calculator", "calculate"),
    "ValueCountsCalculator": ("descriptive", "value_counts_calculator", "calculate"),
    "SeasonalDecomposition": ("time_series", "seasonal_decomposition", "decompose"),
}

HEADER_ADD = '''
from analyze_data.analyzers.backends.pl_utils import ensure_frame, dtypes_dict, describe_dict
from analyze_data.analyzers.backends.stats_runner import run_statistics
'''


def patch_inspectors(text: str) -> str:
    replacements = [
        (
            r"class AnalyseDataTypesPolars.*?return \{\"dtypes\":.*?\}",
            '''class AnalyseDataTypesPolars(AbstractAnalyseDataTypes[pl.DataFrame]):
    """Return column dtypes (native Polars)."""

    def analyze(self, **kwargs) -> dict:
        frame = ensure_frame(self._data_frame)
        return {"dtypes": dtypes_dict(frame)}''',
        ),
        (
            r"class AnalyseDataShapePolars.*?return \{\"rows\":.*?\}",
            '''class AnalyseDataShapePolars(AbstractAnalyseDataShape[pl.DataFrame]):
    def analyze(self, **kwargs) -> dict:
        frame = ensure_frame(self._data_frame)
        return {"rows": frame.height, "columns": frame.width}''',
        ),
        (
            r"class AnalyseDataColumnsPolars.*?return \{\"columns\":.*?\}",
            '''class AnalyseDataColumnsPolars(AbstractAnalyseDataColumns[pl.DataFrame]):
    def analyze(self, **kwargs) -> dict:
        frame = ensure_frame(self._data_frame)
        return {"columns": frame.columns}''',
        ),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, count=1, flags=re.DOTALL)
    return text


def patch_calculators(text: str) -> str:
    for cls, (domain, key, method) in CALC_MAP.items():
        # Pattern: SomethingCalculator().method(data) or .calculate(...)
        text = re.sub(
            rf"{cls}\(\)\.(\w+)\(([^)]+)\)",
            rf'run_statistics("{domain}", "{key}", self._data_frame, method="{method}", \2)',
            text,
        )
    # Remove to_pandas conversion lines
    text = re.sub(
        r"\s*self\._data_frame = self\._data_frame\.to_pandas\(\).*?\n",
        "\n",
        text,
    )
    return text


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "from analyze_data.analyzers.backends.stats_runner import run_statistics" not in text:
        insert_at = text.find("from analyze_data.analyzers.backends.abstract_analyzers import")
        if insert_at == -1:
            insert_at = 0
        else:
            insert_at = text.find("\n", insert_at) + 1
        text = text[:insert_at] + HEADER_ADD + text[insert_at:]
    text = patch_inspectors(text)
    text = patch_calculators(text)
    PATH.write_text(text, encoding="utf-8")
    print("patched", PATH)


if __name__ == "__main__":
    main()
