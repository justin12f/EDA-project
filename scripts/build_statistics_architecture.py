#!/usr/bin/env python3
"""Generate statistics abstract + pandas/polars/spark backends from legacy modules."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "statistics"
SKIP_TOP = {"core", "__pycache__"}

POLARS_SIMPLE = {
    "mean": "pl.col(column).mean()",
    "median": "pl.col(column).median()",
    "std": "pl.col(column).std()",
    "variance": "pl.col(column).var()",
    "standard_deviation": "pl.col(column).std()",
    "min": "pl.col(column).min()",
    "max": "pl.col(column).max()",
}


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def factory_class_name(domain: str) -> str:
    parts = domain.split("_")
    return "".join(p.title() for p in parts) + "StatisticsFactory"


def discover_domains() -> list[str]:
    return sorted(
        p.name
        for p in STATS.iterdir()
        if p.is_dir() and p.name not in SKIP_TOP and not p.name.startswith("_")
    )


def module_files(domain: str) -> list[Path]:
    d = STATS / domain
    skip_names = {"abstract.py", "factory.py"}
    return sorted(
        p
        for p in d.glob("*.py")
        if p.name not in skip_names and not p.name.startswith("_")
    )


def parse_classes(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.endswith("Result"):
            continue
        methods = [
            i.name
            for i in node.body
            if isinstance(i, ast.FunctionDef) and not i.name.startswith("_")
        ]
        if methods and node.name[0].isupper():
            out[node.name] = methods
    return out


def gen_abstract(cname: str, methods: list[str]) -> str:
    lines = [f"class Abstract{cname}(ABC, Generic[T]):", ""]
    for m in methods:
        lines += [
            "    @abstractmethod",
            f"    def {m}(self, data: T, column: str, **kwargs: Any) -> Any: ...",
            "",
        ]
    return "\n".join(lines)


def gen_pandas(cname: str, mod_alias: str, methods: list[str]) -> str:
    lines = [
        f"class {cname}Pandas(Abstract{cname}[pd.DataFrame]):",
        "    def __init__(self) -> None:",
        f"        self._legacy = {mod_alias}.{cname}()",
        "",
    ]
    for m in methods:
        lines += [
            f"    def {m}(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:",
            "        arr = column_to_numpy(data, column)",
            f"        return self._legacy.{m}(arr, **kwargs)",
            "",
        ]
    return "\n".join(lines)


def gen_polars(cname: str, methods: list[str], simple: str) -> str:
    lines = [
        f"class {cname}Polars(Abstract{cname}[pl.DataFrame]):",
        "    def __init__(self) -> None:",
        f"        self._pandas = {cname}Pandas()",
        "",
    ]
    for m in methods:
        if m == "calculate" and simple in POLARS_SIMPLE:
            expr = POLARS_SIMPLE[simple]
            lines += [
                "    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:",
                "        frame = eager(data)",
                f"        return float(frame.select({expr}).item())",
                "",
            ]
        elif m == "classify":
            lines += [
                "    def classify(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:",
                "        frame = eager(data)",
                "        s = numeric_series(frame, column)",
                "        if s.len() < 8:",
                "            raise ValueError('Need at least 8 samples for classify')",
                "        skew = float(s.skew())",
                "        kurt = float(s.kurtosis())",
                "        label = 'symmetric' if abs(skew) < 0.5 else 'skewed'",
                "        return {",
                '            "classification_label": label,',
                '            "skewness": skew,',
                '            "kurtosis": kurt,',
                '            "is_bimodal": False,',
                '            "recommended_transformation": "log1p" if skew > 1 else "none",',
                "        }",
                "",
            ]
        else:
            lines += [
                f"    def {m}(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:",
                f"        return self._pandas.{m}(data, column, **kwargs)",
                "",
            ]
    return "\n".join(lines)


def gen_spark(cname: str, methods: list[str], simple: str) -> str:
    lines = [
        f"class {cname}Spark(Abstract{cname}[SparkDataFrame]):",
        "    def __init__(self) -> None:",
        f"        self._pandas = {cname}Pandas()",
        "",
    ]
    spark_fn = {
        "mean": "F.mean",
        "std": "F.stddev",
        "variance": "F.variance",
        "min": "F.min",
        "max": "F.max",
    }
    for m in methods:
        if m == "calculate" and simple == "median":
            lines += [
                "    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:",
                '        row = data.select(F.expr(f"percentile_approx(`{column}`, 0.5)").alias("v")).collect()[0]',
                '        return float(row["v"])',
                "",
            ]
        elif m == "calculate" and simple in spark_fn:
            fn = spark_fn[simple]
            lines += [
                "    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:",
                f'        row = data.select({fn}(column).alias("v")).collect()[0]',
                '        return float(row["v"])',
                "",
            ]
        else:
            lines += [
                f"    def {m}(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:",
                f"        return self._pandas.{m}(data, column, **kwargs)",
                "",
            ]
    return "\n".join(lines)


def build_domain(domain: str) -> None:
    files = module_files(domain)
    entries: list[tuple[str, str, list[str]]] = []
    for path in files:
        for cname, methods in parse_classes(path).items():
            entries.append((path.stem, cname, methods))

    if not entries:
        return

    abstract_body = [
        f'"""Abstract statistics contracts — domain `{domain}`."""',
        "from __future__ import annotations",
        "from abc import ABC, abstractmethod",
        "from typing import Any, Generic, TypeVar",
        "T = TypeVar('T')",
        "",
    ]
    for _, cname, methods in entries:
        abstract_body.append(gen_abstract(cname, methods))
    (STATS / domain / "abstract.py").write_text("\n".join(abstract_body), encoding="utf-8")

    backends = STATS / domain / "backends"
    backends.mkdir(exist_ok=True)

    pandas = [
        f'"""Pandas statistics backends — `{domain}`."""',
        "from __future__ import annotations",
        "from typing import Any",
        "import pandas as pd",
        "from statistics.core.frame_extract import column_to_numpy",
        f"from statistics.{domain}.abstract import *",
        "",
    ]
    polars = [
        f'"""Polars statistics backends — `{domain}`."""',
        "from __future__ import annotations",
        "from typing import Any",
        "import polars as pl",
        "from pyspark.sql import DataFrame as SparkDataFrame",
        "from pyspark.sql import functions as F",
        "from statistics.core.polars_frame import eager, numeric_series",
        f"from statistics.{domain}.abstract import *",
        "",
    ]
    spark = [
        f'"""Spark statistics backends — `{domain}`."""',
        "from __future__ import annotations",
        "from typing import Any",
        "from pyspark.sql import DataFrame as SparkDataFrame",
        "from pyspark.sql import functions as F",
        f"from statistics.{domain}.abstract import *",
        "",
    ]

    mod_aliases = {}
    for mod_stem, cname, methods in entries:
        alias = f"_mod_{mod_stem}"
        if mod_stem not in mod_aliases:
            mod_aliases[mod_stem] = alias
            pandas.append(f"import statistics.{domain}.{mod_stem} as {alias}")
    pandas.append("")

    for mod_stem, cname, methods in entries:
        alias = mod_aliases[mod_stem]
        pandas.append(gen_pandas(cname, alias, methods))
        simple = camel_to_snake(cname.replace("Calculator", "").replace("Classifier", ""))
        polars.append(gen_polars(cname, methods, simple))
        spark.append(gen_spark(cname, methods, simple))

    polars.insert(7, f"from statistics.{domain}.backends import pandas_impl")
    polars.insert(8, f"from statistics.{domain}.backends.pandas_impl import *")
    polars.insert(9, "")
    spark.insert(7, f"from statistics.{domain}.backends import pandas_impl")
    spark.insert(8, f"from statistics.{domain}.backends.pandas_impl import *")
    spark.insert(9, "")

    (backends / "pandas_impl.py").write_text("\n".join(pandas), encoding="utf-8")
    (backends / "polars_impl.py").write_text("\n".join(polars), encoding="utf-8")
    (backends / "spark_impl.py").write_text("\n".join(spark), encoding="utf-8")

    fn = factory_class_name(domain)
    reg = [
        f'"""Factory — domain `{domain}`."""',
        "from __future__ import annotations",
        "from typing import Any",
        "from core.abstract_factory import RegistryFactory",
        "",
        f"class {fn}(RegistryFactory[str, Any]):",
        "    pass",
        "",
        "def _register() -> None:",
        f"    from statistics.{domain}.backends import pandas_impl as p",
        f"    from statistics.{domain}.backends import polars_impl as pl",
        f"    from statistics.{domain}.backends import spark_impl as sp",
        "",
    ]
    for _, cname, _ in entries:
        key = camel_to_snake(cname)
        reg.append(f'    {fn}.register("{key}", "pandas", p.{cname}Pandas)')
        reg.append(f'    {fn}.register("{key}", "polars", pl.{cname}Polars)')
        reg.append(f'    {fn}.register("{key}", "spark", sp.{cname}Spark)')
    reg += ["", "_register()", ""]
    (STATS / domain / "factory.py").write_text("\n".join(reg), encoding="utf-8")


def build_registry() -> None:
    lines = [
        '"""Unified statistics factory registry."""',
        "from __future__ import annotations",
        "from typing import Any",
        "from core.abstract_factory import RegistryFactory",
        "",
        "class StatisticsRegistry(RegistryFactory[str, Any]):",
        '    """Keys: `{domain}.{calculator}`"""',
        "",
        "DOMAIN_FACTORIES: dict[str, type] = {}",
        "",
        "def _register_all() -> None:",
    ]
    for domain in discover_domains():
        fn = factory_class_name(domain)
        lines.append(f"    from statistics.{domain}.factory import {fn}")
        lines.append(f"    DOMAIN_FACTORIES[{domain!r}] = {fn}")
        lines.append(f"    for key in {fn}.registered_keys():")
        lines.append("        for backend in ('pandas', 'polars', 'spark'):")
        lines.append(f"            if {fn}.is_registered(key, backend):")
        lines.append(
            f"                StatisticsRegistry.register('{domain}.' + key, backend, {fn}.get_class(key, backend))"
        )
    lines += ["", "_register_all()", ""]
    (STATS / "registry.py").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for d in discover_domains():
        print("domain:", d)
        build_domain(d)
    build_registry()
    print("complete")


if __name__ == "__main__":
    main()
