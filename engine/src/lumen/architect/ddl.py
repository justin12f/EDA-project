"""Turning a SchemaSpec into SQL text.

This module is the only place in Lumen that composes SQL from data, which
makes it the only place that needs injection review. `PostgresManager.
execute_query(query: str)` — the orphan from the pre-SaaS codebase — is the
shape this exists to replace: it took whatever string it was handed.
"""

from __future__ import annotations

import re

from lumen.architect.spec import ColumnSpec, SchemaSpec, SpecError

# Postgres truncates identifiers at 63 *bytes*, not characters. Truncating
# by character count leaves a multibyte name the server still rejects.
MAX_IDENTIFIER_BYTES = 63

_VALID = re.compile(r"[a-z_][a-z0-9_]*")

# Not the full reserved list — the subset a real dataset actually collides
# with. A reserved name would work if quoted, but forcing a customer to
# write "select" in every query is a papercut worth one underscore.
_RESERVED = frozenset(
    {
        "all", "and", "any", "as", "asc", "both", "case", "cast", "check",
        "column", "constraint", "create", "default", "desc", "distinct", "do",
        "else", "end", "except", "false", "for", "foreign", "from", "grant",
        "group", "having", "in", "initially", "intersect", "into", "join",
        "leading", "limit", "not", "null", "offset", "on", "only", "or",
        "order", "primary", "references", "select", "session_user", "some",
        "table", "then", "to", "trailing", "true", "union", "unique", "user",
        "using", "when", "where", "with",
    }
)


def _truncate_bytes(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def sanitize_identifier(raw: str, *, taken: set[str] | None = None) -> str:
    """A safe Postgres identifier derived from arbitrary text.

    Deliberately lossy and deterministic: the same input always produces the
    same output, and anything that cannot produce a valid identifier raises
    rather than falling back. A caller that hits `SpecError` here has a bug —
    this is not an escape hatch for exotic names.
    """
    lowered = raw.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")

    if not collapsed:
        raise SpecError(f"identifier {raw!r} cannot be sanitised into a valid name")

    # A leading digit is legal in the source data and illegal in Postgres.
    if collapsed[0].isdigit():
        collapsed = f"col_{collapsed}"

    if collapsed in _RESERVED:
        collapsed = f"{collapsed}_col"

    collapsed = _truncate_bytes(collapsed, MAX_IDENTIFIER_BYTES)

    if taken is not None and collapsed in taken:
        base = collapsed
        suffix = 2
        while True:
            # Re-truncate: the suffix must fit inside the limit too.
            room = MAX_IDENTIFIER_BYTES - len(f"_{suffix}".encode("utf-8"))
            candidate = f"{_truncate_bytes(base, room)}_{suffix}"
            if candidate not in taken:
                collapsed = candidate
                break
            suffix += 1

    # The result is about to be interpolated into DDL. If it does not match,
    # that is a bug in the logic above, not a case to handle gracefully.
    if not _VALID.fullmatch(collapsed):
        raise SpecError(f"sanitising {raw!r} produced invalid identifier {collapsed!r}")

    return collapsed


def _quote(identifier: str) -> str:
    """Wrap an already-sanitised identifier in double quotes.

    Sanitisation has guaranteed the value matches ^[a-z_][a-z0-9_]*$, so
    there is no embedded quote to escape. Quoting anyway is belt-and-braces
    against a future caller that skips sanitisation.
    """
    if not _VALID.fullmatch(identifier):
        raise SpecError(f"refusing to quote unsanitised identifier {identifier!r}")
    return f'"{identifier}"'


def _render_type(column: ColumnSpec) -> str:
    if column.type_arg:
        return f"{column.sql_type.value}({column.type_arg})"
    return column.sql_type.value


def render_ddl(spec: SchemaSpec, schema: str) -> list[str]:
    """The statements that create `spec` inside `schema`.

    Returned as a list rather than one blob because they run inside a single
    transaction and a driver that prepares statements cannot take several at
    once — the same constraint every migration in this repo works around.

    Only enforced foreign keys produce DDL. An observed relationship is
    recorded in the spec and drawn in the diagram, but the database is never
    told something the data does not support.
    """
    spec.validate()

    statements = [f"CREATE SCHEMA IF NOT EXISTS {_quote(schema)}"]

    for table in spec.tables:
        columns = ",\n".join(
            f"  {_quote(c.name)} {_render_type(c)}" + ("" if c.nullable else " NOT NULL")
            for c in table.columns
        )
        statements.append(
            f"CREATE TABLE IF NOT EXISTS {_quote(schema)}.{_quote(table.name)} (\n"
            f"{columns}\n)"
        )

        if table.primary_key:
            key_columns = ", ".join(_quote(c) for c in table.primary_key)
            statements.append(
                f"ALTER TABLE {_quote(schema)}.{_quote(table.name)} "
                f"ADD CONSTRAINT {_quote(f'{table.name}_pkey')} "
                f"PRIMARY KEY ({key_columns})"
            )

    for fk in spec.foreign_keys:
        if not fk.enforced:
            continue
        statements.append(
            f"ALTER TABLE {_quote(schema)}.{_quote(fk.from_table)} "
            f"ADD CONSTRAINT {_quote(f'{fk.from_table}_{fk.from_column}_fkey')} "
            f"FOREIGN KEY ({_quote(fk.from_column)}) "
            f"REFERENCES {_quote(schema)}.{_quote(fk.to_table)} ({_quote(fk.to_column)}) "
            f"DEFERRABLE INITIALLY IMMEDIATE"
        )

    return statements


def render_replace(spec: SchemaSpec, schema: str, table: str) -> list[str]:
    """Clear one table for a full-snapshot reload (D6).

    DELETE rather than TRUNCATE: TRUNCATE cannot be deferred, so a parent
    table referenced by a child would fail immediately even inside a
    transaction. DELETE respects DEFERRABLE, which is why the constraints
    were declared that way in the first place.
    """
    known = {t.name for t in spec.tables}
    if table not in known:
        raise SpecError(f"cannot replace unknown table '{table}'")
    return [
        "SET CONSTRAINTS ALL DEFERRED",
        f"DELETE FROM {_quote(schema)}.{_quote(table)}",
    ]
