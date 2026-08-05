"""Rewrite top-level engine imports to the lumen namespace. Idempotent."""
from __future__ import annotations

import re
import sys
from pathlib import Path

PACKAGES = (
    "agents", "algorithms", "analyze_data", "api", "core", "database",
    "data_cleaning", "evaluation", "models", "model_tools", "parsers",
    "preproccesing", "readers", "statistics",
)

FROM_RE = re.compile(rf"^(\s*)from\s+({'|'.join(PACKAGES)})(\.|\s)", re.M)
IMPORT_RE = re.compile(rf"^(\s*)import\s+({'|'.join(PACKAGES)})(\.|\s|$)", re.M)


def rewrite(text: str) -> str:
    text = FROM_RE.sub(r"\1from lumen.\2\3", text)
    text = IMPORT_RE.sub(r"\1import lumen.\2\3", text)
    return text


def main(root: str) -> int:
    changed = 0
    for path in Path(root).rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        original = path.read_text(encoding="utf-8")
        updated = rewrite(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    print(f"rewrote {changed} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "engine/src/lumen"))
