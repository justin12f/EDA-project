"""The Data Architect — designing a real relational schema from data.

Pure and database-free on purpose. Everything here takes data or a spec in
and returns a spec or SQL text out; nothing opens a connection. That split
is what lets the whole design surface be unit-tested in milliseconds, the
same way `lumen.sentinel.baseline` is testable while `lumen_api.baselines`
needs a live database.
"""

from lumen.architect.spec import (
    ColumnSpec,
    Evidence,
    ForeignKeySpec,
    SchemaSpec,
    SpecError,
    SqlType,
    TableSpec,
)

__all__ = [
    "ColumnSpec",
    "Evidence",
    "ForeignKeySpec",
    "SchemaSpec",
    "SpecError",
    "SqlType",
    "TableSpec",
]
