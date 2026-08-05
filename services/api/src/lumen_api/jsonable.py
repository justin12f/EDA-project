"""Coerce arbitrary Python values — numpy scalars, dataclasses-by-repr, whatever
a tool or a pipeline report handed back — into something `json.dumps` accepts.
Anything not already a primitive, dict, list or tuple is stringified rather than
left to raise deep inside a response encoder.
"""

from __future__ import annotations

from typing import Any


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)
