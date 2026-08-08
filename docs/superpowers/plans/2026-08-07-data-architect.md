# Data Architect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agent that plans, creates and administers a real Postgres database per customer, so that uploaded files and connected customer databases become queryable SQL tables with real types, primary keys and enforced foreign keys.

**Architecture:** A pure, database-free engine core (`lumen.architect`) infers a typed `SchemaSpec` from data and renders it to DDL. A separate Postgres instance holds tenant data, one schema and one role per org. Orchestration in the API layer turns a spec into a `Proposal` a human accepts, then executes the DDL. Worker jobs handle staging load, design, and re-ingest.

**Tech Stack:** Python 3.11, uv, polars, DuckDB, SQLAlchemy + asyncpg, Fernet (cryptography), arq, pytest.

**Spec:** `docs/superpowers/specs/2026-08-07-data-architect-design.md` — read it before starting. Its twelve decisions D1–D12 are binding.

## Global Constraints

- **Never emit model-authored SQL.** The agent produces a typed `SchemaSpec`; `render_ddl()` is the only function in the system that composes SQL text.
- **`Backend` stays `Literal["pandas", "polars", "spark"]`.** DuckDB is a read accelerator, never a fourth backend value.
- **Never `DROP COLUMN`.** A column absent at origin is marked `deprecated`, never dropped. `render_migration()` must be incapable of emitting it.
- **Foreign keys are created `DEFERRABLE INITIALLY IMMEDIATE`.** This is what makes full-snapshot replacement possible.
- **Identifiers are validated against `^[a-z_][a-z0-9_]*$` after sanitisation.** A failure raises `SpecError`; it is never an escape hatch.
- **`enforced=True` requires `containment == 1.0`.** Enforcing a partial relationship is the one invariant `SchemaSpec.validate()` exists to protect.
- **The two-session rule.** Tenant data and control-plane data are on different instances; no transaction spans them. The control-plane record is always written **last**.
- **Engine code is pure.** Anything under `engine/src/lumen/architect/` and `engine/src/lumen/datasets/` takes data in and returns data out. No sessions, no I/O beyond a passed-in connection string.
- **Test markers.** Engine tests run in the default suite. Anything needing a live database is `pytestmark = pytest.mark.integration`.
- **Test commands.** `cd engine && uv run pytest -q <path>` · `cd services/api && uv run pytest -q <path>` · `cd services/api && uv run pytest -q -m integration <path>`
- **ruff:** line-length 100, target py311.
- **Git:** never `git add -A` or `git add .` — always an explicit file list. Never stage anything under `docs/architecture/`.
- **Comments explain why, not what.** Match the surrounding density; this codebase argues for its decisions in prose.

---

## File Structure

| File | Responsibility | New? |
|---|---|---|
| `engine/src/lumen/architect/__init__.py` | package exports | new |
| `engine/src/lumen/architect/spec.py` | `SchemaSpec` and friends; validation | new |
| `engine/src/lumen/architect/ddl.py` | identifier sanitisation; DDL rendering | new |
| `engine/src/lumen/architect/infer.py` | type inference, PK selection, FK detection | new |
| `engine/src/lumen/architect/migrate.py` | schema diff and migration rendering | new |
| `engine/src/lumen/architect/adapters/base.py` | `SourceAdapter` protocol, `Discovered*` types | new |
| `engine/src/lumen/architect/adapters/file.py` | files via the existing `ReaderFactory` | new |
| `engine/src/lumen/architect/adapters/postgres.py` | live customer Postgres | new |
| `engine/src/lumen/architect/adapters/mysql.py` | live customer MySQL | new |
| `engine/src/lumen/datasets/sql_read.py` | two-path reader (polars / DuckDB) | new |
| `services/api/src/lumen_api/tenant_db.py` | tenant engine, provisioning, `tenant_session()` | new |
| `services/api/src/lumen_api/credentials.py` | DSN encryption | new |
| `services/api/src/lumen_api/architect.py` | design → propose → apply orchestration | new |
| `services/worker/src/lumen_worker/ingest.py` | staging load and design jobs | new |
| `services/api/src/lumen_api/settings.py` | two new settings | modify |
| `services/api/src/lumen_api/db/session.py` | docstring: the third session entry point | modify |
| `services/api/src/lumen_api/trust.py` | `structural_shape()` cases for the new kinds | modify |
| `services/api/src/lumen_api/proposals.py` | `decide_proposal` dispatch for the new kinds | modify |
| `services/api/src/lumen_api/datasets/store.py` | `resolve()` reads SQL instead of Parquet | modify |
| `services/api/src/lumen_api/sources.py` | enqueue ingestion; stop assuming CSV | modify |
| `engine/src/lumen/database/postgres_manager.py` | orphan — delete | delete |
| `engine/src/lumen/agents/postgres_admin_agent.py` | orphan — delete | delete |

---

# Phase 1 — Pure engine core

No database, no I/O, no async. Every test in this phase runs in the default engine suite in milliseconds.

### Task 1: The `SchemaSpec` contract

**Files:**
- Create: `engine/src/lumen/architect/__init__.py`
- Create: `engine/src/lumen/architect/spec.py`
- Test: `engine/tests/test_architect_spec.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SqlType`, `Evidence`, `ColumnSpec`, `TableSpec`, `ForeignKeySpec`, `SchemaSpec`, `SpecError`. Every later task depends on these exact names.

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_architect_spec.py`:

```python
"""SchemaSpec validation — the invariants every later stage assumes hold."""

from __future__ import annotations

import uuid

import pytest

from lumen.architect.spec import (
    ColumnSpec,
    Evidence,
    ForeignKeySpec,
    SchemaSpec,
    SpecError,
    SqlType,
    TableSpec,
)

SOURCE = uuid.uuid4()


def _table(name: str, *columns: str, pk: tuple[str, ...] | None = None) -> TableSpec:
    return TableSpec(
        name=name,
        source_id=SOURCE,
        columns=tuple(
            ColumnSpec(name=c, source_column=c, sql_type=SqlType.TEXT) for c in columns
        ),
        primary_key=pk,
        pk_rationale="test fixture",
    )


def test_a_minimal_valid_spec_passes():
    SchemaSpec(tables=(_table("orders", "id", "customer_id", pk=("id",)),)).validate()


def test_duplicate_table_names_are_rejected():
    spec = SchemaSpec(tables=(_table("orders", "id"), _table("orders", "id")))
    with pytest.raises(SpecError, match="duplicate table"):
        spec.validate()


def test_duplicate_column_names_within_a_table_are_rejected():
    spec = SchemaSpec(tables=(_table("orders", "id", "id"),))
    with pytest.raises(SpecError, match="duplicate column"):
        spec.validate()


def test_a_primary_key_naming_an_unknown_column_is_rejected():
    spec = SchemaSpec(tables=(_table("orders", "id", pk=("missing",)),))
    with pytest.raises(SpecError, match="primary key"):
        spec.validate()


def test_a_foreign_key_referencing_an_unknown_table_is_rejected():
    spec = SchemaSpec(
        tables=(_table("orders", "id", "customer_id", pk=("id",)),),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=1.0,
                enforced=True,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="unknown table"):
        spec.validate()


def test_a_foreign_key_referencing_an_unknown_column_is_rejected():
    spec = SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="nope",
                to_table="customers",
                to_column="id",
                containment=1.0,
                enforced=True,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="unknown column"):
        spec.validate()


@pytest.mark.parametrize("containment", [-0.1, 1.1])
def test_containment_outside_zero_to_one_is_rejected(containment):
    spec = SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=containment,
                enforced=False,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="containment"):
        spec.validate()


def test_an_enforced_key_with_partial_containment_is_rejected():
    """Decision D5's core invariant: Postgres cannot enforce a relationship
    the data does not fully satisfy, so the spec must not be able to claim it."""
    spec = SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=0.97,
                enforced=True,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    )
    with pytest.raises(SpecError, match="enforced"):
        spec.validate()


def test_partial_containment_is_fine_when_not_enforced():
    SchemaSpec(
        tables=(
            _table("orders", "id", "customer_id", pk=("id",)),
            _table("customers", "id", pk=("id",)),
        ),
        foreign_keys=(
            ForeignKeySpec(
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="id",
                containment=0.97,
                enforced=False,
                evidence=(Evidence.STRUCTURAL,),
                rationale="test",
            ),
        ),
    ).validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_spec.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen.architect'`

- [ ] **Step 3: Write the implementation**

Create `engine/src/lumen/architect/__init__.py`:

```python
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
```

Create `engine/src/lumen/architect/spec.py`:

```python
"""The typed schema contract.

The agent never emits SQL. It emits one of these, and `ddl.render_ddl`
turns it into statements. That indirection is the whole safety argument:
a model can propose a shape, but it cannot propose syntax.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class SpecError(ValueError):
    """A spec that would produce invalid or dishonest DDL."""


class SqlType(StrEnum):
    """A closed vocabulary. Types are never interpolated from model output —
    a value not in this enum cannot reach `render_ddl`."""

    TEXT = "text"
    VARCHAR = "varchar"
    INTEGER = "integer"
    BIGINT = "bigint"
    NUMERIC = "numeric"
    DOUBLE = "double precision"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    TIMESTAMPTZ = "timestamptz"
    UUID = "uuid"
    JSONB = "jsonb"


class Evidence(StrEnum):
    """Why we believe a relationship exists, strongest first.

    DECLARED outranks everything: it means the customer's own database
    already declares the constraint, so there is nothing to infer.
    """

    DECLARED = "declared"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    NAMING = "naming"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    source_column: str
    sql_type: SqlType
    # "255" for varchar(255); "12,2" for numeric(12,2). None for types that
    # take no argument, which is most of them.
    type_arg: str | None = None
    nullable: bool = True
    # Absent at origin but retained. D7 forbids DROP COLUMN outright, which
    # is what keeps almost every migration reversible.
    deprecated: bool = False


@dataclass(frozen=True)
class TableSpec:
    name: str
    source_id: uuid.UUID
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] | None = None
    pk_rationale: str = ""
    source_table: str | None = None


@dataclass(frozen=True)
class ForeignKeySpec:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    # Fraction of child values present in the parent. 1.0 is the only value
    # that may be enforced — see SchemaSpec.validate.
    containment: float
    enforced: bool
    evidence: tuple[Evidence, ...]
    rationale: str


@dataclass(frozen=True)
class SchemaSpec:
    tables: tuple[TableSpec, ...]
    foreign_keys: tuple[ForeignKeySpec, ...] = ()
    layout: Literal["merged", "namespaced"] = "merged"

    def validate(self) -> None:
        by_name: dict[str, TableSpec] = {}
        for table in self.tables:
            if table.name in by_name:
                raise SpecError(f"duplicate table name '{table.name}'")
            by_name[table.name] = table

            seen: set[str] = set()
            for column in table.columns:
                if column.name in seen:
                    raise SpecError(
                        f"duplicate column '{column.name}' in table '{table.name}'"
                    )
                seen.add(column.name)

            for key_column in table.primary_key or ():
                if key_column not in seen:
                    raise SpecError(
                        f"primary key of '{table.name}' names unknown column '{key_column}'"
                    )

        for fk in self.foreign_keys:
            for side, table_name, column_name in (
                ("from", fk.from_table, fk.from_column),
                ("to", fk.to_table, fk.to_column),
            ):
                table = by_name.get(table_name)
                if table is None:
                    raise SpecError(
                        f"foreign key {side} side references unknown table '{table_name}'"
                    )
                if column_name not in {c.name for c in table.columns}:
                    raise SpecError(
                        f"foreign key {side} side references unknown column "
                        f"'{table_name}.{column_name}'"
                    )

            if not 0.0 <= fk.containment <= 1.0:
                raise SpecError(
                    f"containment {fk.containment} for "
                    f"{fk.from_table}.{fk.from_column} is outside [0, 1]"
                )

            # The invariant D5 exists to protect. Postgres will reject the
            # constraint at CREATE time anyway, but failing here means the
            # spec is honest before it ever reaches a database.
            if fk.enforced and fk.containment < 1.0:
                raise SpecError(
                    f"{fk.from_table}.{fk.from_column} is marked enforced but only "
                    f"{fk.containment:.1%} of its values exist in "
                    f"{fk.to_table}.{fk.to_column}"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_spec.py`
Expected: PASS — 10 passed

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/__init__.py engine/src/lumen/architect/spec.py engine/tests/test_architect_spec.py
git commit -m "feat(architect): the typed schema contract

SchemaSpec is what the agent emits instead of SQL. validate() enforces the
one invariant everything downstream assumes: a foreign key may only be
marked enforced when containment is exactly 1.0, because Postgres cannot
enforce a relationship the data does not fully satisfy and a spec that
claims otherwise is dishonest before it ever reaches a database."
```

---

### Task 2: Identifier sanitisation

This is the single SQL-injection surface in the entire system, which is why it gets its own task and its own reviewer gate rather than riding along with DDL rendering.

**Files:**
- Create: `engine/src/lumen/architect/ddl.py`
- Test: `engine/tests/test_architect_ddl.py`

**Interfaces:**
- Consumes: `SpecError` from `lumen.architect.spec`.
- Produces: `sanitize_identifier(raw: str, *, taken: set[str] | None = None) -> str`.

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_architect_ddl.py`:

```python
"""Identifier sanitisation — the only place in the system where untrusted
text becomes SQL, so the only place that needs injection review."""

from __future__ import annotations

import pytest

from lumen.architect.ddl import sanitize_identifier
from lumen.architect.spec import SpecError


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("orders", "orders"),
        ("Orders", "orders"),
        ("Customer ID", "customer_id"),
        ("customer-id", "customer_id"),
        ("customer..id", "customer_id"),
        ("  spaced  ", "spaced"),
        ("2024_revenue", "col_2024_revenue"),
        ("café", "caf_"),
    ],
)
def test_sanitisation_cases(raw, expected):
    assert sanitize_identifier(raw) == expected


def test_a_reserved_word_is_suffixed_not_quoted_away():
    """Quoting would work, but a column a customer has to write as "select"
    in every query is a papercut we can spend one underscore to avoid."""
    assert sanitize_identifier("select") == "select_col"
    assert sanitize_identifier("ORDER") == "order_col"


def test_collisions_get_a_numeric_suffix():
    taken: set[str] = set()
    first = sanitize_identifier("Customer ID", taken=taken)
    taken.add(first)
    second = sanitize_identifier("customer.id", taken=taken)
    taken.add(second)
    third = sanitize_identifier("CUSTOMER-ID", taken=taken)
    assert [first, second, third] == ["customer_id", "customer_id_2", "customer_id_3"]


def test_a_long_name_is_truncated_to_the_postgres_limit():
    result = sanitize_identifier("a" * 200)
    assert len(result.encode("utf-8")) == 63


def test_truncation_counts_bytes_not_characters():
    """Postgres's limit is 63 bytes. A multibyte name truncated by character
    count would still be rejected by the server."""
    result = sanitize_identifier("ñ" * 100)
    assert len(result.encode("utf-8")) <= 63


def test_a_name_with_nothing_usable_raises():
    with pytest.raises(SpecError, match="cannot be sanitised"):
        sanitize_identifier("!!!")


def test_an_empty_name_raises():
    with pytest.raises(SpecError, match="cannot be sanitised"):
        sanitize_identifier("")


def test_injection_attempts_are_neutralised():
    assert sanitize_identifier('x"; drop table users; --') == "x_drop_table_users_"


def test_every_result_matches_the_validation_pattern():
    import re

    for raw in ["orders", "Customer ID", "2024", "select", "a" * 200, "ñoño"]:
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", sanitize_identifier(raw))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_ddl.py`
Expected: FAIL with `ImportError: cannot import name 'sanitize_identifier'`

- [ ] **Step 3: Write the implementation**

Create `engine/src/lumen/architect/ddl.py`:

```python
"""Turning a SchemaSpec into SQL text.

This module is the only place in Lumen that composes SQL from data, which
makes it the only place that needs injection review. `PostgresManager.
execute_query(query: str)` — the orphan from the pre-SaaS codebase — is the
shape this exists to replace: it took whatever string it was handed.
"""

from __future__ import annotations

import re

from lumen.architect.spec import SpecError

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_ddl.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/ddl.py engine/tests/test_architect_ddl.py
git commit -m "feat(architect): identifier sanitisation

The one place untrusted text becomes SQL, so it gets its own module and its
own tests. Truncation counts bytes rather than characters because Postgres's
63-limit is bytes and a multibyte name cut by character count is still
rejected by the server. A name that cannot yield a valid identifier raises
rather than falling back — reaching that branch is a caller bug, not a case
to absorb."
```

---

### Task 3: Type inference

**Files:**
- Create: `engine/src/lumen/architect/infer.py`
- Test: `engine/tests/test_architect_infer.py`

**Interfaces:**
- Consumes: `SqlType` from `lumen.architect.spec`.
- Produces: `infer_sql_type(dtype: str) -> tuple[SqlType, str | None]`.

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_architect_infer.py`:

```python
"""Deterministic inference — no model call reaches any of this."""

from __future__ import annotations

import pytest

from lumen.architect.infer import infer_sql_type
from lumen.architect.spec import SqlType


@pytest.mark.parametrize(
    "dtype,expected",
    [
        # polars
        ("Int8", SqlType.INTEGER),
        ("Int32", SqlType.INTEGER),
        ("Int64", SqlType.BIGINT),
        ("UInt64", SqlType.BIGINT),
        ("Float32", SqlType.DOUBLE),
        ("Float64", SqlType.DOUBLE),
        ("Boolean", SqlType.BOOLEAN),
        ("Utf8", SqlType.TEXT),
        ("String", SqlType.TEXT),
        ("Date", SqlType.DATE),
        ("Decimal(12, 2)", SqlType.NUMERIC),
        # pandas
        ("int64", SqlType.BIGINT),
        ("int32", SqlType.INTEGER),
        ("float64", SqlType.DOUBLE),
        ("bool", SqlType.BOOLEAN),
        ("object", SqlType.TEXT),
        ("category", SqlType.TEXT),
    ],
)
def test_scalar_dtypes(dtype, expected):
    assert infer_sql_type(dtype)[0] is expected


@pytest.mark.parametrize(
    "dtype",
    [
        "Datetime(time_unit='us', time_zone=None)",
        "datetime64[ns]",
    ],
)
def test_naive_datetimes_map_to_timestamp(dtype):
    assert infer_sql_type(dtype)[0] is SqlType.TIMESTAMP


@pytest.mark.parametrize(
    "dtype",
    [
        "Datetime(time_unit='us', time_zone='UTC')",
        "datetime64[ns, UTC]",
    ],
)
def test_aware_datetimes_map_to_timestamptz(dtype):
    """Dropping the zone would silently reinterpret every timestamp, which is
    the kind of corruption nobody notices until a report is wrong."""
    assert infer_sql_type(dtype)[0] is SqlType.TIMESTAMPTZ


@pytest.mark.parametrize("dtype", ["List(Int64)", "Struct({'a': Int64})"])
def test_nested_types_map_to_jsonb(dtype):
    assert infer_sql_type(dtype)[0] is SqlType.JSONB


def test_decimal_carries_its_precision_and_scale():
    assert infer_sql_type("Decimal(12, 2)") == (SqlType.NUMERIC, "12,2")


def test_a_type_without_an_argument_returns_none():
    assert infer_sql_type("Int64") == (SqlType.BIGINT, None)


def test_an_unknown_dtype_falls_back_to_text():
    """Falling back is right: a column we cannot type is still a column the
    customer wants to see, and text loses nothing that was in the file."""
    assert infer_sql_type("SomeFutureType") == (SqlType.TEXT, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_infer.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen.architect.infer'`

- [ ] **Step 3: Write the implementation**

Create `engine/src/lumen/architect/infer.py`:

```python
"""Deterministic schema inference.

Nothing here calls a model. That is the same discipline `lumen.sentinel.
drift` states for detection and `lumen.sentinel.baseline` for calibration:
the expensive, non-reproducible tier is reserved for judgment, and a type
mapping is not judgment. The model's only job downstream is to make the
names readable and write the rationale prose.
"""

from __future__ import annotations

import re

from lumen.architect.spec import SqlType

# Exact dtype strings, checked before the prefix rules below. Ordering
# matters: "int64" must not be matched by a naive "int" prefix rule that
# would call it INTEGER.
_EXACT: dict[str, SqlType] = {
    "int8": SqlType.INTEGER,
    "int16": SqlType.INTEGER,
    "int32": SqlType.INTEGER,
    "int64": SqlType.BIGINT,
    "uint8": SqlType.INTEGER,
    "uint16": SqlType.INTEGER,
    "uint32": SqlType.BIGINT,
    "uint64": SqlType.BIGINT,
    "float32": SqlType.DOUBLE,
    "float64": SqlType.DOUBLE,
    "boolean": SqlType.BOOLEAN,
    "bool": SqlType.BOOLEAN,
    "utf8": SqlType.TEXT,
    "string": SqlType.TEXT,
    "str": SqlType.TEXT,
    "object": SqlType.TEXT,
    "category": SqlType.TEXT,
    "date": SqlType.DATE,
    "time": SqlType.TEXT,
    "binary": SqlType.TEXT,
    "null": SqlType.TEXT,
}

_DECIMAL = re.compile(r"decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def infer_sql_type(dtype: str) -> tuple[SqlType, str | None]:
    """Map a polars or pandas dtype string to a Postgres type.

    Returns `(type, type_arg)` where `type_arg` carries the parenthesised
    part for the two types that take one — `numeric(p,s)` and `varchar(n)`.
    An unrecognised dtype falls back to TEXT rather than raising: a column
    we cannot type is still a column the customer wants, and text loses
    nothing the file contained.
    """
    normalised = dtype.strip().lower()

    decimal = _DECIMAL.search(normalised)
    if decimal:
        return SqlType.NUMERIC, f"{decimal.group(1)},{decimal.group(2)}"

    if normalised.startswith("datetime"):
        # Both dialects spell the zone differently — polars as
        # time_zone='UTC', pandas as datetime64[ns, UTC] — but the question
        # is the same: is there a zone at all. Dropping one would silently
        # reinterpret every value in the column.
        aware = "time_zone='" in normalised or re.search(r",\s*[a-z/_+\-0-9]+\]", normalised)
        return (SqlType.TIMESTAMPTZ if aware else SqlType.TIMESTAMP), None

    if normalised.startswith(("list(", "array(", "struct(")):
        return SqlType.JSONB, None

    if normalised.startswith("duration"):
        return SqlType.TEXT, None

    exact = _EXACT.get(normalised)
    if exact is not None:
        return exact, None

    if normalised.startswith("uuid"):
        return SqlType.UUID, None

    return SqlType.TEXT, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_infer.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/infer.py engine/tests/test_architect_infer.py
git commit -m "feat(architect): dtype to Postgres type inference

Deterministic, no model call — the same discipline lumen.sentinel states
for detection. The one case worth naming: a timezone-aware datetime maps to
timestamptz and a naive one to timestamp, because collapsing the two
silently reinterprets every value in the column, which is the kind of
corruption nobody notices until a report is wrong."
```

---

### Task 4: Primary key selection

**Files:**
- Modify: `engine/src/lumen/architect/infer.py` (append)
- Test: `engine/tests/test_architect_infer.py` (append)

**Interfaces:**
- Consumes: `duplicate_counts` and `null_rates` from `lumen.datasets.materialize` (both already exist — read that file).
- Produces: `select_primary_key(frame, backend: str, columns: list[str]) -> tuple[tuple[str, ...] | None, str]`.

- [ ] **Step 1: Write the failing test**

Append to `engine/tests/test_architect_infer.py`:

```python
# ── primary key selection ───────────────────────────────────────────────

import polars as pl  # noqa: E402

from lumen.architect.infer import select_primary_key  # noqa: E402


def test_a_column_named_id_wins_over_other_candidates():
    frame = pl.DataFrame({"code": ["a", "b"], "id": [1, 2]})
    key, rationale = select_primary_key(frame, "polars", ["code", "id"])
    assert key == ("id",)
    assert "id" in rationale


def test_an_id_suffixed_column_wins_when_there_is_no_bare_id():
    frame = pl.DataFrame({"name": ["a", "b"], "order_id": [1, 2]})
    key, _ = select_primary_key(frame, "polars", ["name", "order_id"])
    assert key == ("order_id",)


def test_the_leftmost_unique_column_wins_when_no_name_hints():
    frame = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    key, _ = select_primary_key(frame, "polars", ["a", "b"])
    assert key == ("a",)


def test_a_column_with_duplicates_is_not_a_candidate():
    frame = pl.DataFrame({"id": [1, 1], "code": ["x", "y"]})
    key, _ = select_primary_key(frame, "polars", ["id", "code"])
    assert key == ("code",)


def test_a_column_with_nulls_is_not_a_candidate():
    frame = pl.DataFrame({"id": [1, None], "code": ["x", "y"]})
    key, _ = select_primary_key(frame, "polars", ["id", "code"])
    assert key == ("code",)


def test_no_viable_key_returns_none_with_an_explanation():
    """The rationale is shown to a human deciding whether to accept the
    schema, so it has to read as a reason, not a stack trace."""
    frame = pl.DataFrame({"a": [1, 1], "b": [2, 2]})
    key, rationale = select_primary_key(frame, "polars", ["a", "b"])
    assert key is None
    assert "no column" in rationale.lower()


def test_an_empty_frame_returns_none():
    frame = pl.DataFrame({"a": []})
    key, _ = select_primary_key(frame, "polars", ["a"])
    assert key is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_infer.py -k primary_key`
Expected: FAIL with `ImportError: cannot import name 'select_primary_key'`

- [ ] **Step 3: Write the implementation**

Append to `engine/src/lumen/architect/infer.py`:

```python
def select_primary_key(
    frame: Any, backend: str, columns: list[str]
) -> tuple[tuple[str, ...] | None, str]:
    """Choose a single-column primary key, or explain why there isn't one.

    Single-column only in v1. A composite key is a modelling judgment — which
    combination is *the* identity, rather than merely unique together — and
    D2's conservative posture says a human makes that call, not an inference
    rule. The returned rationale is shown to that human, so it is written as
    an explanation rather than a trace.
    """
    if not columns:
        return None, "The table has no columns."

    if row_count(frame, backend) == 0:
        return None, "The table is empty, so no column can be shown to be unique."

    rates = null_rates(frame, backend)
    duplicates = duplicate_counts(frame, backend, list(columns))

    candidates = [
        column
        for column in columns
        if duplicates.get(column, 1) == 0 and rates.get(column, 1.0) == 0.0
    ]
    if not candidates:
        return None, (
            "No column is both unique and complete, so no primary key was declared. "
            "Every column either repeats a value or has gaps."
        )

    for preferred, why in (
        (lambda c: c == "id", "it is named 'id'"),
        (lambda c: c.endswith("_id"), "its name identifies it as a key"),
    ):
        for column in candidates:
            if preferred(column):
                return (column,), (
                    f"'{column}' is unique and complete across every row, and {why}."
                )

    column = candidates[0]
    return (column,), (
        f"'{column}' is unique and complete across every row, and is the first "
        f"such column in the table."
    )
```

Add to that file's imports at the top:

```python
from typing import Any

from lumen.datasets.materialize import duplicate_counts, null_rates, row_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_infer.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/infer.py engine/tests/test_architect_infer.py
git commit -m "feat(architect): primary key selection

Reuses duplicate_counts and null_rates rather than recomputing uniqueness —
materialize.py already answers both questions in one pass per backend.

Single-column only in v1, deliberately. Which combination of columns is
*the* identity, as opposed to merely unique together, is a modelling
judgment, and D2's conservative posture leaves that to a human. The
rationale string is written as an explanation because it is shown to the
person deciding whether to accept the schema."
```

---

### Task 5: Foreign key detection

**Files:**
- Modify: `engine/src/lumen/architect/infer.py` (append)
- Test: `engine/tests/test_architect_infer.py` (append)

**Interfaces:**
- Consumes: `column_values` from `lumen.datasets.materialize`; `ForeignKeySpec`, `Evidence`, `TableSpec` from `lumen.architect.spec`.
- Produces: `detect_foreign_keys(frames: dict[str, Any], backend: str, tables: tuple[TableSpec, ...], semantic_pairs: list[tuple[str, str, str, str]] | None = None) -> list[ForeignKeySpec]`.

- [ ] **Step 1: Write the failing test**

Append to `engine/tests/test_architect_infer.py`:

```python
# ── foreign key detection ───────────────────────────────────────────────

import uuid  # noqa: E402

from lumen.architect.infer import detect_foreign_keys  # noqa: E402
from lumen.architect.spec import ColumnSpec, Evidence, SqlType, TableSpec  # noqa: E402

_SRC = uuid.uuid4()


def _spec(name: str, columns: list[str], pk: str | None) -> TableSpec:
    return TableSpec(
        name=name,
        source_id=_SRC,
        columns=tuple(
            ColumnSpec(name=c, source_column=c, sql_type=SqlType.TEXT) for c in columns
        ),
        primary_key=(pk,) if pk else None,
        pk_rationale="",
    )


def _pair():
    frames = {
        "customers": pl.DataFrame({"id": ["c1", "c2", "c3"]}),
        "orders": pl.DataFrame({"id": ["o1", "o2"], "customer_id": ["c1", "c2"]}),
    }
    tables = (
        _spec("customers", ["id"], "id"),
        _spec("orders", ["id", "customer_id"], "id"),
    )
    return frames, tables


def test_total_containment_produces_an_enforced_key():
    frames, tables = _pair()
    keys = detect_foreign_keys(frames, "polars", tables)
    assert len(keys) == 1
    key = keys[0]
    assert (key.from_table, key.from_column) == ("orders", "customer_id")
    assert (key.to_table, key.to_column) == ("customers", "id")
    assert key.containment == 1.0
    assert key.enforced is True


def test_partial_containment_produces_an_observed_key():
    frames, tables = _pair()
    frames["orders"] = pl.DataFrame(
        {"id": [f"o{i}" for i in range(100)],
         "customer_id": ["c1"] * 97 + ["ghost1", "ghost2", "ghost3"]}
    )
    keys = detect_foreign_keys(frames, "polars", tables)
    assert len(keys) == 1
    assert keys[0].enforced is False
    assert keys[0].containment == pytest.approx(0.5, abs=0.5)


def test_containment_below_the_floor_is_dropped_entirely():
    frames, tables = _pair()
    frames["orders"] = pl.DataFrame(
        {"id": ["o1", "o2"], "customer_id": ["nope1", "nope2"]}
    )
    assert detect_foreign_keys(frames, "polars", tables) == []


def test_a_matching_name_adds_naming_evidence():
    frames, tables = _pair()
    key = detect_foreign_keys(frames, "polars", tables)[0]
    assert Evidence.NAMING in key.evidence
    assert Evidence.STRUCTURAL in key.evidence


def test_a_semantic_pair_adds_semantic_evidence():
    """This is how ADR-0009's canonical_entities reach the engine without
    the engine knowing what a canonical entity is."""
    frames, tables = _pair()
    key = detect_foreign_keys(
        frames, "polars", tables,
        semantic_pairs=[("orders", "customer_id", "customers", "id")],
    )[0]
    assert Evidence.SEMANTIC in key.evidence


def test_a_column_that_is_not_the_parents_primary_key_is_not_a_target():
    frames = {
        "customers": pl.DataFrame({"id": ["c1"], "region": ["north"]}),
        "orders": pl.DataFrame({"id": ["o1"], "region": ["north"]}),
    }
    tables = (
        _spec("customers", ["id", "region"], "id"),
        _spec("orders", ["id", "region"], "id"),
    )
    keys = detect_foreign_keys(frames, "polars", tables)
    assert all(k.to_column == "id" for k in keys)


def test_no_self_referencing_keys_in_v1():
    frames = {"nodes": pl.DataFrame({"id": ["a", "b"], "parent": ["a", "a"]})}
    tables = (_spec("nodes", ["id", "parent"], "id"),)
    assert detect_foreign_keys(frames, "polars", tables) == []


def test_an_empty_child_column_is_skipped():
    frames = {
        "customers": pl.DataFrame({"id": ["c1"]}),
        "orders": pl.DataFrame({"id": [], "customer_id": []}, schema={"id": pl.Utf8, "customer_id": pl.Utf8}),
    }
    tables = (
        _spec("customers", ["id"], "id"),
        _spec("orders", ["id", "customer_id"], "id"),
    )
    assert detect_foreign_keys(frames, "polars", tables) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_infer.py -k foreign`
Expected: FAIL with `ImportError: cannot import name 'detect_foreign_keys'`

- [ ] **Step 3: Write the implementation**

Append to `engine/src/lumen/architect/infer.py`:

```python
# Below this share of child values present in the parent, a match is
# coincidence rather than a relationship. Chosen conservative on purpose:
# a missed key costs a line on a diagram, a false one costs a human's trust
# in every other line.
CONTAINMENT_FLOOR = 0.95


def _names_suggest_a_key(child_column: str, parent_table: str, parent_column: str) -> bool:
    singular = parent_table[:-1] if parent_table.endswith("s") else parent_table
    return child_column in {
        f"{singular}_{parent_column}",
        f"{parent_table}_{parent_column}",
        parent_column if child_column != parent_column else "",
    } or child_column == f"{singular}_id"


def detect_foreign_keys(
    frames: dict[str, Any],
    backend: str,
    tables: tuple[TableSpec, ...],
    semantic_pairs: list[tuple[str, str, str, str]] | None = None,
) -> list[ForeignKeySpec]:
    """Find relationships between tables by value containment.

    The classic algorithm: a child column references a parent when every one
    of its values exists in the parent's primary key. `column_values` and the
    set arithmetic are what `impact.py::_match_rate` already does for
    ADR-0010's cross-source match rate, applied to a different question.

    `semantic_pairs` carries ADR-0009's canonical entities in as
    `(child_table, child_column, parent_table, parent_column)` tuples. The
    engine never learns what a canonical entity is; it just weighs the hint.
    """
    semantic = set(semantic_pairs or ())
    parents = [
        (table.name, table.primary_key[0])
        for table in tables
        if table.primary_key and len(table.primary_key) == 1
    ]

    found: list[ForeignKeySpec] = []
    for child in tables:
        child_frame = frames.get(child.name)
        if child_frame is None:
            continue

        for parent_table, parent_column in parents:
            # A self-reference is a real pattern (org charts, threaded
            # comments) but it needs its own handling on replace and its own
            # UI treatment, so v1 declines rather than half-supporting it.
            if parent_table == child.name:
                continue
            parent_frame = frames.get(parent_table)
            if parent_frame is None:
                continue

            parent_values = column_values(parent_frame, backend, parent_column)
            if not parent_values:
                continue

            for column in child.columns:
                if column.name == parent_column and child.name == parent_table:
                    continue
                if child.primary_key and column.name in child.primary_key:
                    continue

                child_values = column_values(child_frame, backend, column.name)
                if not child_values:
                    continue

                containment = len(child_values & parent_values) / len(child_values)
                if containment < CONTAINMENT_FLOOR:
                    continue

                evidence = [Evidence.STRUCTURAL]
                if _names_suggest_a_key(column.name, parent_table, parent_column):
                    evidence.append(Evidence.NAMING)
                if (child.name, column.name, parent_table, parent_column) in semantic:
                    evidence.append(Evidence.SEMANTIC)

                found.append(
                    ForeignKeySpec(
                        from_table=child.name,
                        from_column=column.name,
                        to_table=parent_table,
                        to_column=parent_column,
                        containment=containment,
                        enforced=containment == 1.0,
                        evidence=tuple(evidence),
                        rationale=(
                            f"{containment:.1%} of values in "
                            f"{child.name}.{column.name} exist in "
                            f"{parent_table}.{parent_column}, which is that "
                            f"table's primary key."
                        ),
                    )
                )
    return found
```

Extend that file's imports:

```python
from lumen.architect.spec import Evidence, ForeignKeySpec, SqlType, TableSpec
from lumen.datasets.materialize import column_values, duplicate_counts, null_rates, row_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_infer.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/infer.py engine/tests/test_architect_infer.py
git commit -m "feat(architect): foreign key detection by containment

The classic algorithm — a child references a parent when its values are a
subset of the parent's primary key — built on column_values, which
impact.py::_match_rate already uses for the same set arithmetic against a
different question.

Three evidence sources combine: structural containment always, naming when
the column name identifies its parent, and semantic when ADR-0009's
canonical entities are passed in. The engine never learns what a canonical
entity is; semantic_pairs is how that signal crosses the boundary.

The 0.95 floor is conservative by choice. A missed key costs a line on a
diagram; a false one costs the customer's trust in every other line."
```

---

### Task 6: DDL rendering

**Files:**
- Modify: `engine/src/lumen/architect/ddl.py` (append)
- Test: `engine/tests/test_architect_ddl.py` (append)

**Interfaces:**
- Consumes: `SchemaSpec`, `TableSpec`, `SqlType` from `lumen.architect.spec`; `sanitize_identifier` from Task 2.
- Produces: `render_ddl(spec: SchemaSpec, schema: str) -> list[str]`, `render_replace(spec: SchemaSpec, schema: str, table: str) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Append to `engine/tests/test_architect_ddl.py`:

```python
# ── DDL rendering ───────────────────────────────────────────────────────

import uuid  # noqa: E402

from lumen.architect.ddl import render_ddl, render_replace  # noqa: E402
from lumen.architect.spec import (  # noqa: E402
    ColumnSpec,
    Evidence,
    ForeignKeySpec,
    SchemaSpec,
    SqlType,
    TableSpec,
)

_SRC = uuid.uuid4()


def _customers() -> TableSpec:
    return TableSpec(
        name="customers",
        source_id=_SRC,
        columns=(
            ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT, nullable=False),
            ColumnSpec(name="amount", source_column="amount", sql_type=SqlType.NUMERIC, type_arg="12,2"),
        ),
        primary_key=("id",),
        pk_rationale="",
    )


def _orders() -> TableSpec:
    return TableSpec(
        name="orders",
        source_id=_SRC,
        columns=(
            ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT, nullable=False),
            ColumnSpec(name="customer_id", source_column="customer_id", sql_type=SqlType.TEXT),
        ),
        primary_key=("id",),
        pk_rationale="",
    )


def _fk(enforced: bool, containment: float) -> ForeignKeySpec:
    return ForeignKeySpec(
        from_table="orders",
        from_column="customer_id",
        to_table="customers",
        to_column="id",
        containment=containment,
        enforced=enforced,
        evidence=(Evidence.STRUCTURAL,),
        rationale="",
    )


def test_create_schema_comes_first():
    statements = render_ddl(SchemaSpec(tables=(_customers(),)), "tenant_abc")
    assert statements[0] == 'CREATE SCHEMA IF NOT EXISTS "tenant_abc"'


def test_a_table_renders_with_quoted_identifiers_and_real_types():
    statements = render_ddl(SchemaSpec(tables=(_customers(),)), "tenant_abc")
    assert statements[1] == (
        'CREATE TABLE IF NOT EXISTS "tenant_abc"."customers" (\n'
        '  "id" text NOT NULL,\n'
        '  "amount" numeric(12,2)\n'
        ')'
    )


def test_a_primary_key_renders_as_its_own_constraint():
    statements = render_ddl(SchemaSpec(tables=(_customers(),)), "tenant_abc")
    assert any(
        s == 'ALTER TABLE "tenant_abc"."customers" '
             'ADD CONSTRAINT "customers_pkey" PRIMARY KEY ("id")'
        for s in statements
    )


def test_an_enforced_key_is_deferrable():
    """SET CONSTRAINTS ALL DEFERRED only works on constraints declared
    DEFERRABLE, and D6's replace-in-one-transaction depends on it."""
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(True, 1.0),))
    statements = render_ddl(spec, "tenant_abc")
    fk = [s for s in statements if "FOREIGN KEY" in s]
    assert len(fk) == 1
    assert fk[0] == (
        'ALTER TABLE "tenant_abc"."orders" '
        'ADD CONSTRAINT "orders_customer_id_fkey" '
        'FOREIGN KEY ("customer_id") REFERENCES "tenant_abc"."customers" ("id") '
        'DEFERRABLE INITIALLY IMMEDIATE'
    )


def test_an_observed_key_emits_no_ddl_at_all():
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(False, 0.97),))
    statements = render_ddl(spec, "tenant_abc")
    assert not any("FOREIGN KEY" in s for s in statements)


def test_a_deprecated_column_is_still_created():
    """D7 keeps deprecated columns; dropping them is what it forbids."""
    table = TableSpec(
        name="orders",
        source_id=_SRC,
        columns=(
            ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT, nullable=False),
            ColumnSpec(name="legacy", source_column="legacy", sql_type=SqlType.TEXT, deprecated=True),
        ),
        primary_key=("id",),
        pk_rationale="",
    )
    statements = render_ddl(SchemaSpec(tables=(table,)), "tenant_abc")
    assert '"legacy" text' in statements[1]


def test_render_replace_defers_constraints_before_deleting():
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(True, 1.0),))
    statements = render_replace(spec, "tenant_abc", "customers")
    assert statements == [
        "SET CONSTRAINTS ALL DEFERRED",
        'DELETE FROM "tenant_abc"."customers"',
    ]


def test_no_rendered_statement_ever_contains_drop():
    spec = SchemaSpec(tables=(_customers(), _orders()), foreign_keys=(_fk(True, 1.0),))
    for statement in render_ddl(spec, "tenant_abc") + render_replace(spec, "tenant_abc", "customers"):
        assert "DROP" not in statement.upper()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_ddl.py -k render`
Expected: FAIL with `ImportError: cannot import name 'render_ddl'`

- [ ] **Step 3: Write the implementation**

Append to `engine/src/lumen/architect/ddl.py`:

```python
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
```

Extend that file's imports:

```python
from lumen.architect.spec import ColumnSpec, SchemaSpec, SpecError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_ddl.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/ddl.py engine/tests/test_architect_ddl.py
git commit -m "feat(architect): DDL rendering

Two decisions worth recording. Foreign keys are DEFERRABLE INITIALLY
IMMEDIATE because D6's full-snapshot replacement runs SET CONSTRAINTS ALL
DEFERRED inside one transaction, and that only works on constraints
declared deferrable.

And render_replace uses DELETE rather than TRUNCATE: TRUNCATE cannot be
deferred, so clearing a parent table referenced by a child would fail
immediately even inside a transaction. DELETE respects the deferral, which
is the whole reason the constraints were declared that way.

Only enforced keys emit DDL. An observed relationship is drawn in the
diagram but never asserted to the database."
```

---

### Task 7: Migration classification and rendering

**Files:**
- Create: `engine/src/lumen/architect/migrate.py`
- Test: `engine/tests/test_architect_migrate.py`

**Interfaces:**
- Consumes: `SchemaSpec`, `ColumnSpec`, `SqlType` from `lumen.architect.spec`; `_quote` and `_render_type` from `lumen.architect.ddl`.
- Produces: `MigrationStep`, `MigrationPlan`, `classify_migration(old, new) -> MigrationPlan`, `render_migration(plan, schema) -> list[str]`.

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_architect_migrate.py`:

```python
"""Schema evolution — what the agent may do alone, and what it may never do."""

from __future__ import annotations

import uuid

import pytest

from lumen.architect.migrate import classify_migration, render_migration
from lumen.architect.spec import ColumnSpec, SchemaSpec, SqlType, TableSpec

_SRC = uuid.uuid4()


def _spec(*columns: tuple[str, SqlType]) -> SchemaSpec:
    return SchemaSpec(
        tables=(
            TableSpec(
                name="orders",
                source_id=_SRC,
                columns=tuple(
                    ColumnSpec(name=n, source_column=n, sql_type=t) for n, t in columns
                ),
                primary_key=None,
                pk_rationale="",
            ),
        )
    )


def test_a_new_column_is_additive_and_reversible():
    plan = classify_migration(_spec(("id", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT), ("email", SqlType.TEXT)))
    assert [s.kind for s in plan.steps] == ["add_column"]
    assert plan.reversible is True


def test_a_widening_type_change_is_reversible():
    plan = classify_migration(_spec(("n", SqlType.INTEGER)), _spec(("n", SqlType.BIGINT)))
    assert [s.kind for s in plan.steps] == ["widen_type"]
    assert plan.reversible is True


@pytest.mark.parametrize(
    "old,new",
    [
        (SqlType.INTEGER, SqlType.BIGINT),
        (SqlType.INTEGER, SqlType.NUMERIC),
        (SqlType.VARCHAR, SqlType.TEXT),
        (SqlType.DOUBLE, SqlType.TEXT),
    ],
)
def test_widening_pairs(old, new):
    plan = classify_migration(_spec(("n", old)), _spec(("n", new)))
    assert plan.reversible is True


def test_a_narrowing_type_change_is_not_reversible():
    plan = classify_migration(_spec(("n", SqlType.TEXT)), _spec(("n", SqlType.INTEGER)))
    assert [s.kind for s in plan.steps] == ["narrow_type"]
    assert plan.reversible is False


def test_a_column_absent_at_origin_is_deprecated_not_dropped():
    plan = classify_migration(_spec(("id", SqlType.TEXT), ("legacy", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT)))
    assert [s.kind for s in plan.steps] == ["deprecate_column"]
    assert plan.reversible is True


def test_an_unchanged_schema_produces_an_empty_plan():
    plan = classify_migration(_spec(("id", SqlType.TEXT)), _spec(("id", SqlType.TEXT)))
    assert plan.steps == ()
    assert plan.reversible is True


def test_render_migration_never_emits_drop_column():
    """D7's hard rule. The test exists because this is the single line that,
    if it regressed, would silently make the product destructive."""
    plan = classify_migration(_spec(("id", SqlType.TEXT), ("legacy", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT)))
    for statement in render_migration(plan, "tenant_abc"):
        assert "DROP" not in statement.upper()


def test_render_migration_adds_a_column():
    plan = classify_migration(_spec(("id", SqlType.TEXT)),
                              _spec(("id", SqlType.TEXT), ("email", SqlType.TEXT)))
    assert render_migration(plan, "tenant_abc") == [
        'ALTER TABLE "tenant_abc"."orders" ADD COLUMN IF NOT EXISTS "email" text'
    ]


def test_render_migration_widens_a_type():
    plan = classify_migration(_spec(("n", SqlType.INTEGER)), _spec(("n", SqlType.BIGINT)))
    assert render_migration(plan, "tenant_abc") == [
        'ALTER TABLE "tenant_abc"."orders" ALTER COLUMN "n" TYPE bigint'
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_migrate.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen.architect.migrate'`

- [ ] **Step 3: Write the implementation**

Create `engine/src/lumen/architect/migrate.py`:

```python
"""Schema evolution — the "administers" half of the Architect.

D7 gates evolution on reversibility rather than on a list of allowed
operations, and the reason is ADR-0017 §3: irreversibility is a hard
ceiling that no amount of earned trust may raise. If dropping a column were
possible, a computer-authored migration could never be auto-applied at all,
because every schema change would carry an irreversible branch. Refusing to
drop is what keeps the whole mechanism eligible for autonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lumen.architect.ddl import _quote, _render_type
from lumen.architect.spec import SchemaSpec, SqlType

# A change is reversible when the old values still fit the new type. These
# are the pairs where that holds without inspecting a single row.
_WIDENING: frozenset[tuple[SqlType, SqlType]] = frozenset(
    {
        (SqlType.INTEGER, SqlType.BIGINT),
        (SqlType.INTEGER, SqlType.NUMERIC),
        (SqlType.INTEGER, SqlType.DOUBLE),
        (SqlType.BIGINT, SqlType.NUMERIC),
        (SqlType.BIGINT, SqlType.DOUBLE),
        (SqlType.NUMERIC, SqlType.DOUBLE),
        (SqlType.VARCHAR, SqlType.TEXT),
        (SqlType.DATE, SqlType.TIMESTAMP),
        (SqlType.TIMESTAMP, SqlType.TIMESTAMPTZ),
    }
)

MigrationKind = Literal[
    "add_column", "widen_type", "deprecate_column", "narrow_type", "add_fk", "drop_fk"
]


@dataclass(frozen=True)
class MigrationStep:
    kind: MigrationKind
    table: str
    column: str | None
    detail: str
    reversible: bool


@dataclass(frozen=True)
class MigrationPlan:
    steps: tuple[MigrationStep, ...] = ()

    @property
    def reversible(self) -> bool:
        return all(step.reversible for step in self.steps)


def _is_widening(old: SqlType, new: SqlType) -> bool:
    # Anything becoming text is lossless — text holds every representation.
    return new is SqlType.TEXT or (old, new) in _WIDENING


def classify_migration(old: SchemaSpec, new: SchemaSpec) -> MigrationPlan:
    """Diff two specs into steps, each labelled reversible or not."""
    steps: list[MigrationStep] = []
    old_tables = {t.name: t for t in old.tables}

    for table in new.tables:
        previous = old_tables.get(table.name)
        if previous is None:
            continue  # a brand-new table is a create, not a migration

        old_columns = {c.name: c for c in previous.columns}
        new_columns = {c.name: c for c in table.columns}

        for name, column in new_columns.items():
            before = old_columns.get(name)
            if before is None:
                steps.append(
                    MigrationStep(
                        kind="add_column",
                        table=table.name,
                        column=name,
                        detail=f"new column '{name}' ({column.sql_type.value})",
                        reversible=True,
                    )
                )
            elif before.sql_type is not column.sql_type:
                widening = _is_widening(before.sql_type, column.sql_type)
                steps.append(
                    MigrationStep(
                        kind="widen_type" if widening else "narrow_type",
                        table=table.name,
                        column=name,
                        detail=(
                            f"'{name}' changes from {before.sql_type.value} "
                            f"to {column.sql_type.value}"
                        ),
                        reversible=widening,
                    )
                )

        for name in old_columns:
            if name not in new_columns:
                steps.append(
                    MigrationStep(
                        kind="deprecate_column",
                        table=table.name,
                        column=name,
                        detail=f"'{name}' is no longer present at origin",
                        reversible=True,
                    )
                )

    return MigrationPlan(steps=tuple(steps))


def render_migration(plan: MigrationPlan, schema: str) -> list[str]:
    """Statements for a plan.

    `deprecate_column` renders nothing. That is the point: the column stays,
    the flag lives in the spec, and no path through this function can emit
    DROP COLUMN.
    """
    statements: list[str] = []
    for step in plan.steps:
        if step.kind == "add_column":
            column_type = step.detail.rsplit("(", 1)[-1].rstrip(")")
            statements.append(
                f"ALTER TABLE {_quote(schema)}.{_quote(step.table)} "
                f"ADD COLUMN IF NOT EXISTS {_quote(step.column)} {column_type}"
            )
        elif step.kind in ("widen_type", "narrow_type"):
            new_type = step.detail.rsplit(" to ", 1)[-1]
            statements.append(
                f"ALTER TABLE {_quote(schema)}.{_quote(step.table)} "
                f"ALTER COLUMN {_quote(step.column)} TYPE {new_type}"
            )
    return statements
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_migrate.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/migrate.py engine/tests/test_architect_migrate.py
git commit -m "feat(architect): schema evolution gated by reversibility

D7 gates on reversibility rather than on an allow-list of operations,
because of ADR-0017 §3: irreversibility is a ceiling no earned trust may
raise. If DROP COLUMN were reachable, every schema change would carry an
irreversible branch and a migration could never be auto-applied at all.
Refusing to drop is what keeps the mechanism eligible for autonomy.

deprecate_column therefore renders nothing — the column stays and the flag
lives in the spec. A test asserts no rendered statement ever contains DROP,
because that single line regressing is what would silently make the product
destructive."
```

- [ ] **Step 6: Run the whole engine suite before moving on**

Run: `cd engine && uv run pytest -q`
Expected: PASS — the pre-existing 209 tests plus everything added in Phase 1.

---

# Phase 2 — Tenant infrastructure

The second Postgres instance and the isolation that makes it safe for an agent to write DDL.

### Task 8: Settings, the tenant engine, and `tenant_session()`

**Files:**
- Modify: `services/api/src/lumen_api/settings.py` (add two settings, following the existing field style)
- Create: `services/api/src/lumen_api/tenant_db.py`
- Modify: `services/api/src/lumen_api/db/session.py` (module docstring only)
- Test: `services/api/tests/test_tenant_db.py`

**Interfaces:**
- Consumes: `get_settings()` from `lumen_api.settings`.
- Produces: `tenant_schema_name(org_id) -> str`, `tenant_raw_schema_name(org_id) -> str`, `tenant_role_name(org_id) -> str`, `get_tenant_engine()`, `tenant_session(org_id)`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_tenant_db.py`:

```python
"""Tenant identifier naming — pure, so it runs in the default suite.

Provisioning and isolation need a live instance and live in
test_tenant_isolation.py behind the integration marker.
"""

from __future__ import annotations

import uuid

from lumen_api.tenant_db import (
    tenant_raw_schema_name,
    tenant_role_name,
    tenant_schema_name,
)

ORG = uuid.UUID("f7930655-ed4d-40f0-9a8d-21cd99bf468a")


def test_the_schema_name_strips_dashes():
    assert tenant_schema_name(ORG) == "tenant_f7930655ed4d40f09a8d21cd99bf468a"


def test_the_raw_schema_is_the_schema_plus_a_suffix():
    assert tenant_raw_schema_name(ORG) == tenant_schema_name(ORG) + "_raw"


def test_the_role_is_the_schema_plus_a_suffix():
    assert tenant_role_name(ORG) == tenant_schema_name(ORG) + "_role"


def test_every_identifier_fits_the_postgres_limit():
    """63 bytes. The role name is the longest of the three, so if it fits,
    they all do — but assert each one rather than reasoning about it."""
    for name in (tenant_schema_name(ORG), tenant_raw_schema_name(ORG), tenant_role_name(ORG)):
        assert len(name.encode("utf-8")) <= 63


def test_names_are_deterministic():
    assert tenant_schema_name(ORG) == tenant_schema_name(ORG)


def test_different_orgs_get_different_names():
    assert tenant_schema_name(ORG) != tenant_schema_name(uuid.uuid4())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q tests/test_tenant_db.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen_api.tenant_db'`

- [ ] **Step 3: Write the implementation**

Add to `services/api/src/lumen_api/settings.py`, following the field style already in that file:

```python
    # ADR-0024: customer data lives on its own Postgres instance, never in
    # the Supabase database that holds organizations, api_keys and
    # subscriptions. Unset means the Architect is disabled, which is the
    # correct state for a checkout that has not provisioned one.
    tenant_database_url: SecretStr | None = None
    # Above this row count a table is read through DuckDB rather than
    # polars (spec §3.7). A compute choice only — results are identical
    # either side, which a parity test enforces.
    duckdb_row_threshold: int = 5_000_000

    @property
    def has_tenant_db(self) -> bool:
        return self.tenant_database_url is not None
```

Create `services/api/src/lumen_api/tenant_db.py`:

```python
"""The tenant Postgres instance.

Separate from Supabase on purpose (spec D1). The control plane —
organizations, memberships, subscriptions, api_keys, proposals — stays
there; this instance holds nothing but customer data. The security argument
is structural rather than permissive: an agent writing DDL here has no
connection to a control-plane table, so the blast radius is bounded by what
the instance *contains*, not by what a REVOKE forbids.

Per-org roles are still required. They solve a different problem: keeping
tenants out of each other's schemas *within* this instance.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from lumen_api.settings import get_settings

_engines: dict[int, AsyncEngine] = {}


def tenant_schema_name(org_id: uuid.UUID) -> str:
    """`tenant_<32 hex>` — 39 bytes, comfortably inside Postgres's 63."""
    return f"tenant_{org_id.hex}"


def tenant_raw_schema_name(org_id: uuid.UUID) -> str:
    """Staging. Permanent, not a temporary buffer: the raw-data browser
    reads it while a schema is awaiting review, and a failed promotion must
    be retryable without re-downloading the origin."""
    return f"{tenant_schema_name(org_id)}_raw"


def tenant_role_name(org_id: uuid.UUID) -> str:
    return f"{tenant_schema_name(org_id)}_role"


def get_tenant_engine() -> AsyncEngine:
    """Cached per event loop, for the reason `db/session.py` documents at
    length: an asyncpg connection belongs to the loop that opened it, and a
    module-level pool works in production but breaks under pytest-asyncio,
    which gives every test a fresh loop.
    """
    settings = get_settings()
    if not settings.has_tenant_db:
        raise RuntimeError(
            "TENANT_DATABASE_URL is not configured — the Data Architect is disabled."
        )

    try:
        key = id(asyncio.get_running_loop())
    except RuntimeError:
        key = 0

    engine = _engines.get(key)
    if engine is None:
        engine = create_async_engine(
            settings.tenant_database_url.get_secret_value(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            pool_recycle=1800,
        )
        _engines[key] = engine
    return engine


async def dispose_tenant_engines() -> None:
    for engine in list(_engines.values()):
        await engine.dispose()
    _engines.clear()


@asynccontextmanager
async def tenant_session(org_id: uuid.UUID) -> AsyncIterator[AsyncSession]:
    """A session scoped to one org's schemas by Postgres grants.

    Cannot write `proposals`, `artifact_dependencies` or anything else in
    the control plane — those are on a different instance, so no transaction
    spans both. Callers that need both use two sessions in sequence and
    write the control-plane record LAST (spec §3.2).
    """
    schema = tenant_schema_name(org_id)
    factory = async_sessionmaker(get_tenant_engine(), expire_on_commit=False, class_=AsyncSession)

    async with factory() as session:
        await session.begin()
        await session.execute(text(f'SET LOCAL ROLE "{tenant_role_name(org_id)}"'))
        await session.execute(
            text(f'SET LOCAL search_path = "{schema}", "{tenant_raw_schema_name(org_id)}"')
        )
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

Extend the module docstring of `services/api/src/lumen_api/db/session.py` — it enumerates every session entry point, and a third one now exists:

```python
    tenant_session(org_id)  lives in lumen_api.tenant_db, not this module,
                            because it connects to a different instance
                            entirely. Customer data only; it can reach no
                            table defined in this database.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q tests/test_tenant_db.py`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/tenant_db.py services/api/src/lumen_api/settings.py services/api/src/lumen_api/db/session.py services/api/tests/test_tenant_db.py
git commit -m "feat(tenant): the tenant Postgres instance and its session helper

Customer data gets its own instance (D1). The control plane stays in
Supabase, so an agent writing DDL has no connection to organizations,
api_keys or subscriptions at all — the blast radius is bounded by what the
instance contains rather than by what a grant forbids.

The engine is cached per event loop for the same reason db/session.py
documents: an asyncpg connection belongs to the loop that opened it, and a
module-level pool passes in production but fails under pytest-asyncio.

An unset TENANT_DATABASE_URL disables the Architect rather than failing
obscurely, which is the right state for a checkout with no instance yet."
```

---

### Task 9: Provisioning and the isolation tests

The most important task in this plan. If the isolation tests do not exist, the design is not implemented.

**Files:**
- Modify: `services/api/src/lumen_api/tenant_db.py` (append)
- Test: `services/api/tests/test_tenant_isolation.py`

**Interfaces:**
- Consumes: `tenant_schema_name`, `tenant_raw_schema_name`, `tenant_role_name`, `get_tenant_engine` from Task 8.
- Produces: `ensure_tenant_schema(org_id: uuid.UUID) -> None`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_tenant_isolation.py`:

```python
"""Tenant isolation against the live instance.

The two tests that matter here are (b) and (c). Everything else in this
plan is a feature; these are the reason the feature is safe to ship.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen_api.settings import get_settings
from lumen_api.tenant_db import (
    ensure_tenant_schema,
    get_tenant_engine,
    tenant_schema_name,
    tenant_session,
)

pytestmark = pytest.mark.integration

# Skip rather than error when no instance is configured — a developer
# without one should see a clear skip, not a confusing connection failure.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db,
        reason="TENANT_DATABASE_URL is not configured",
    ),
]


@pytest.fixture
def org_a() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def org_b() -> uuid.UUID:
    return uuid.uuid4()


async def _drop(org_id: uuid.UUID) -> None:
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{tenant_schema_name(org_id)}" CASCADE'))
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{tenant_schema_name(org_id)}_raw" CASCADE'))
        await conn.execute(text(f'DROP ROLE IF EXISTS "{tenant_schema_name(org_id)}_role"'))


async def test_provisioning_is_idempotent(org_a):
    try:
        await ensure_tenant_schema(org_a)
        await ensure_tenant_schema(org_a)  # must not raise

        engine = get_tenant_engine()
        async with engine.connect() as conn:
            count = (
                await conn.execute(
                    text(
                        "select count(*) from information_schema.schemata "
                        "where schema_name = :name"
                    ),
                    {"name": tenant_schema_name(org_a)},
                )
            ).scalar_one()
        assert count == 1
    finally:
        await _drop(org_a)


async def test_one_org_cannot_read_another_orgs_schema(org_a, org_b):
    """The critical test. Asserting on the RAISE matters: a version that
    merely returned zero rows would pass against a broken implementation
    that granted access to an empty schema."""
    try:
        await ensure_tenant_schema(org_a)
        await ensure_tenant_schema(org_b)

        engine = get_tenant_engine()
        async with engine.begin() as conn:
            await conn.execute(
                text(f'CREATE TABLE "{tenant_schema_name(org_b)}".secrets (v text)')
            )
            await conn.execute(
                text(f"INSERT INTO \"{tenant_schema_name(org_b)}\".secrets VALUES ('classified')")
            )

        with pytest.raises(Exception) as caught:
            async with tenant_session(org_a) as db:
                await db.execute(text(f'SELECT * FROM "{tenant_schema_name(org_b)}".secrets'))
        assert "permission denied" in str(caught.value).lower()
    finally:
        await _drop(org_a)
        await _drop(org_b)


async def test_the_tenant_instance_holds_no_control_plane_table():
    """Verifies the separation as a fact about the instance rather than
    trusting that it was configured correctly."""
    engine = get_tenant_engine()
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "select table_name from information_schema.tables "
                    "where table_name in "
                    "('organizations', 'api_keys', 'subscriptions', 'memberships', 'proposals')"
                )
            )
        ).scalars().all()
    assert rows == [], f"control-plane tables found on the tenant instance: {rows}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q -m integration tests/test_tenant_isolation.py`
Expected: FAIL with `ImportError: cannot import name 'ensure_tenant_schema'`

- [ ] **Step 3: Write the implementation**

Append to `services/api/src/lumen_api/tenant_db.py`:

```python
async def ensure_tenant_schema(org_id: uuid.UUID) -> None:
    """Create this org's role and schemas if they are missing.

    Lazy and idempotent rather than part of signup, because orgs already
    exist without these objects and a migration cannot create a role per
    future org. Runs at the head of every ingestion job. Safe under
    concurrency: every statement is IF NOT EXISTS or wrapped in the
    duplicate_object guard every migration in this repo already uses.
    """
    schema = tenant_schema_name(org_id)
    raw = tenant_raw_schema_name(org_id)
    role = tenant_role_name(org_id)

    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"DO $$ BEGIN CREATE ROLE \"{role}\" NOLOGIN; "
                f"EXCEPTION WHEN duplicate_object THEN NULL; END $$"
            )
        )
        for name in (schema, raw):
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{name}" AUTHORIZATION "{role}"'))
            await conn.execute(text(f'GRANT ALL ON SCHEMA "{name}" TO "{role}"'))
        # The connecting role must be able to SET ROLE to it.
        await conn.execute(text(f'GRANT "{role}" TO CURRENT_USER'))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q -m integration tests/test_tenant_isolation.py`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/tenant_db.py services/api/tests/test_tenant_isolation.py
git commit -m "feat(tenant): idempotent provisioning and the isolation tests

Provisioning is lazy rather than part of signup: orgs already exist without
these objects, and a migration cannot create a role for an org that does
not exist yet. It runs at the head of every ingestion job and is safe under
concurrency.

Two tests carry the design. One asserts that org A's role RAISES on org B's
schema — asserting on the raise rather than on an empty result, because a
broken implementation granting access to an empty schema would pass the
weaker assertion. The other asserts the tenant instance contains no
control-plane table at all, verifying the separation as a fact about the
instance rather than trusting it was configured correctly."
```

---

### Task 10: DSN encryption

**Files:**
- Create: `services/api/src/lumen_api/credentials.py`
- Modify: `services/api/pyproject.toml` (add `cryptography`)
- Modify: `services/api/src/lumen_api/settings.py` (one setting)
- Test: `services/api/tests/test_credentials.py`

**Interfaces:**
- Consumes: `get_settings()`.
- Produces: `encrypt_dsn(plain: str) -> str`, `decrypt_dsn(stored: str) -> str`, `CredentialError`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_credentials.py`:

```python
"""DSN encryption. `data_sources.dsn_encrypted` has carried the comment
"Encrypted before it is written" since the first migration with nothing
behind it — this is that."""

from __future__ import annotations

import pytest

from lumen_api.credentials import CredentialError, decrypt_dsn, encrypt_dsn

DSN = "postgresql://acme:hunter2@db.acme.example:5432/production"


def test_round_trip():
    assert decrypt_dsn(encrypt_dsn(DSN)) == DSN


def test_the_plaintext_never_appears_in_the_ciphertext():
    stored = encrypt_dsn(DSN)
    for fragment in ("hunter2", "acme", "db.acme.example", "production"):
        assert fragment not in stored


def test_the_stored_form_carries_a_key_version():
    """So a future rotation needs no migration — the prefix says which key
    encrypted this row."""
    assert encrypt_dsn(DSN).startswith("v1:")


def test_a_tampered_token_raises():
    stored = encrypt_dsn(DSN)
    tampered = stored[:-4] + ("aaaa" if not stored.endswith("aaaa") else "bbbb")
    with pytest.raises(CredentialError):
        decrypt_dsn(tampered)


def test_an_unknown_key_version_raises_clearly():
    with pytest.raises(CredentialError, match="version"):
        decrypt_dsn("v99:whatever")


def test_a_malformed_value_raises():
    with pytest.raises(CredentialError):
        decrypt_dsn("not-even-prefixed")


def test_two_encryptions_of_the_same_dsn_differ():
    """Fernet includes a random IV, so identical inputs must not produce
    identical ciphertext — otherwise the column leaks which customers share
    a database."""
    assert encrypt_dsn(DSN) != encrypt_dsn(DSN)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q tests/test_credentials.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen_api.credentials'`

- [ ] **Step 3: Write the implementation**

Add `"cryptography>=43"` to the `dependencies` list in `services/api/pyproject.toml`, then run `cd services/api && uv sync`.

Add to `settings.py`:

```python
    # Fernet key for customer database credentials. Distinct from every
    # Supabase key on purpose: this one encrypts data belonging to the
    # customer, not access to our own infrastructure.
    credential_encryption_key: SecretStr | None = None
```

Create `services/api/src/lumen_api/credentials.py`:

```python
"""Encryption for customer database credentials.

`data_sources.dsn_encrypted` has carried the comment "Never returned by any
read endpoint. Encrypted before it is written" since the first migration,
with no implementation behind it. This is that implementation.

The plaintext DSN exists only inside an adapter at connection time. It never
enters a response, a log, an agent's context, or a proposal's spec.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from lumen_api.settings import get_settings

# The stored form is "<version>:<token>". Carrying the version means a key
# rotation is a code change plus a lazy re-encrypt, never a migration that
# has to rewrite every row at once.
_CURRENT_VERSION = "v1"


class CredentialError(RuntimeError):
    """A credential that cannot be encrypted or decrypted."""


def _cipher() -> Fernet:
    settings = get_settings()
    if settings.credential_encryption_key is None:
        raise CredentialError(
            "CREDENTIAL_ENCRYPTION_KEY is not configured — customer database "
            "sources cannot be connected without it."
        )
    try:
        return Fernet(settings.credential_encryption_key.get_secret_value().encode())
    except (ValueError, TypeError) as exc:
        raise CredentialError("CREDENTIAL_ENCRYPTION_KEY is not a valid Fernet key") from exc


def encrypt_dsn(plain: str) -> str:
    token = _cipher().encrypt(plain.encode()).decode()
    return f"{_CURRENT_VERSION}:{token}"


def decrypt_dsn(stored: str) -> str:
    version, _, token = stored.partition(":")
    if not token:
        raise CredentialError("stored credential is malformed")
    if version != _CURRENT_VERSION:
        raise CredentialError(f"unknown credential key version {version!r}")
    try:
        return _cipher().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise CredentialError("stored credential failed authentication") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q tests/test_credentials.py`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/credentials.py services/api/src/lumen_api/settings.py services/api/pyproject.toml services/api/tests/test_credentials.py
git commit -m "feat(credentials): encryption for customer database DSNs

data_sources.dsn_encrypted has carried the comment 'Encrypted before it is
written' since the first migration with nothing behind it. This is that.

The stored form is v1:<token> so a key rotation is a code change plus a
lazy re-encrypt rather than a migration that must rewrite every row at once.
A test asserts two encryptions of the same DSN differ, because Fernet's
random IV is what stops the column from leaking which customers share a
database host."
```

---

# Phase 3 — Adapters and the read tier

### Task 11: The adapter protocol and `FileAdapter`

**Files:**
- Create: `engine/src/lumen/architect/adapters/__init__.py`
- Create: `engine/src/lumen/architect/adapters/base.py`
- Create: `engine/src/lumen/architect/adapters/file.py`
- Test: `engine/tests/test_architect_adapters.py`

**Interfaces:**
- Consumes: `ReaderFactory` from `lumen.readers.reader_factory`; `infer_sql_type` from Task 3; `SqlType` from Task 1.
- Produces: `DiscoveredColumn`, `DiscoveredTable`, `DiscoveredStructure`, `SourceAdapter`, `FileAdapter`.

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_architect_adapters.py`:

```python
"""Source adapters. The contract is format-agnostic; v1 ships files and two
live databases, and adding a format is a new adapter and nothing else."""

from __future__ import annotations

import polars as pl
import pytest

from lumen.architect.adapters.file import FileAdapter
from lumen.architect.spec import SqlType


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("id,amount,note\n1,10.5,hello\n2,20.0,world\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def parquet_path(tmp_path):
    path = tmp_path / "orders.parquet"
    pl.DataFrame({"id": [1, 2], "amount": [10.5, 20.0]}).write_parquet(path)
    return str(path)


async def test_discover_returns_one_table_named_after_the_file(csv_path):
    structure = await FileAdapter(csv_path).discover()
    assert len(structure.tables) == 1
    assert structure.tables[0].name == "orders"


async def test_a_file_declares_no_keys(csv_path):
    """Nothing in a CSV asserts a primary key or a relationship. Saying so
    explicitly is what lets the diagram distinguish a read constraint from
    an inferred one."""
    table = (await FileAdapter(csv_path).discover()).tables[0]
    assert table.primary_key is None
    assert table.foreign_keys == ()


async def test_discovery_is_marked_undeclared(csv_path):
    assert (await FileAdapter(csv_path).discover()).declared is False


async def test_columns_are_typed_from_the_frame(csv_path):
    table = (await FileAdapter(csv_path).discover()).tables[0]
    types = {c.name: c.sql_type for c in table.columns}
    assert types["id"] is SqlType.BIGINT
    assert types["amount"] is SqlType.DOUBLE
    assert types["note"] is SqlType.TEXT


async def test_parquet_works_through_the_same_adapter(parquet_path):
    structure = await FileAdapter(parquet_path).discover()
    assert {c.name for c in structure.tables[0].columns} == {"id", "amount"}


async def test_read_returns_a_frame(csv_path):
    frame = await FileAdapter(csv_path).read("orders")
    materialised = frame.collect() if hasattr(frame, "collect") else frame
    assert materialised.height == 2


async def test_read_honours_a_limit(csv_path):
    frame = await FileAdapter(csv_path).read("orders", limit=1)
    materialised = frame.collect() if hasattr(frame, "collect") else frame
    assert materialised.height == 1


async def test_an_unsupported_extension_names_what_is_supported(tmp_path):
    path = tmp_path / "data.docx"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="csv"):
        await FileAdapter(str(path)).discover()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_adapters.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen.architect.adapters'`

- [ ] **Step 3: Write the implementation**

Create `engine/src/lumen/architect/adapters/__init__.py`:

```python
"""Source adapters — one interface, any format."""

from lumen.architect.adapters.base import (
    DiscoveredColumn,
    DiscoveredStructure,
    DiscoveredTable,
    SourceAdapter,
)
from lumen.architect.adapters.file import FileAdapter

__all__ = [
    "DiscoveredColumn",
    "DiscoveredStructure",
    "DiscoveredTable",
    "FileAdapter",
    "SourceAdapter",
]
```

Create `engine/src/lumen/architect/adapters/base.py`:

```python
"""What every source must be able to answer.

Two questions: what is your structure, and give me a table. A file infers
its structure; a live database reads its own. `declared` is how the caller
tells those apart, and it is the difference between a diagram showing a
constraint the database actually holds and one showing a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from lumen.architect.spec import SqlType


@dataclass(frozen=True)
class DiscoveredColumn:
    name: str
    sql_type: SqlType
    type_arg: str | None = None
    nullable: bool = True


@dataclass(frozen=True)
class DiscoveredTable:
    name: str
    columns: tuple[DiscoveredColumn, ...]
    primary_key: tuple[str, ...] | None = None
    # (from_column, to_table, to_column) — populated only when the source
    # declares its own constraints.
    foreign_keys: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class DiscoveredStructure:
    tables: tuple[DiscoveredTable, ...]
    declared: bool


@runtime_checkable
class SourceAdapter(Protocol):
    kind: str
    supports_incremental: bool

    async def discover(self) -> DiscoveredStructure: ...

    async def read(self, table: str, limit: int | None = None) -> Any: ...
```

Create `engine/src/lumen/architect/adapters/file.py`:

```python
"""Files, through the reader factory that already exists.

`ReaderFactory` already registers .csv, .parquet, .json, .xlsx and .xls for
polars and pandas. This adapter deliberately reimplements none of that — a
new format is a registration there, not a change here.
"""

from __future__ import annotations

import os
from typing import Any

from lumen.architect.adapters.base import DiscoveredColumn, DiscoveredStructure, DiscoveredTable
from lumen.architect.infer import infer_sql_type
from lumen.datasets.materialize import frame_schema
from lumen.readers.reader_factory import ReaderFactory


class FileAdapter:
    kind = "file"
    # A file is a full snapshot every time; there is no watermark to resume
    # from, so a refresh is always a full reload.
    supports_incremental = False

    def __init__(self, path: str, backend: str = "polars") -> None:
        self._path = path
        self._backend = backend

    @property
    def table_name(self) -> str:
        return os.path.splitext(os.path.basename(self._path))[0]

    async def discover(self) -> DiscoveredStructure:
        frame = ReaderFactory.create(self._path, backend=self._backend).read()
        schema = frame_schema(frame, self._backend)

        columns = []
        for name, dtype in schema.items():
            sql_type, type_arg = infer_sql_type(dtype)
            columns.append(DiscoveredColumn(name=name, sql_type=sql_type, type_arg=type_arg))

        # No primary key and no foreign keys: a file asserts neither. The
        # Architect infers both later, and `declared=False` is what keeps
        # the diagram honest about which is which.
        return DiscoveredStructure(
            tables=(DiscoveredTable(name=self.table_name, columns=tuple(columns)),),
            declared=False,
        )

    async def read(self, table: str, limit: int | None = None) -> Any:
        frame = ReaderFactory.create(self._path, backend=self._backend).read()
        if limit is None:
            return frame
        return frame.head(limit)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_adapters.py`
Expected: PASS — 8 passed

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/adapters/ engine/tests/test_architect_adapters.py
git commit -m "feat(architect): source adapter protocol and the file adapter

Ingestion becomes a contract rather than a code path with a CSV branch
(D9). FileAdapter wraps the ReaderFactory that already registers csv,
parquet, json, xlsx and xls — a new format is a registration there, not a
change here.

The load-bearing field is DiscoveredStructure.declared. A file asserts no
primary key and no relationships, so it discovers as declared=False and
everything is inferred; a live database declares its own constraints and
discovers as True. That flag is the difference between a diagram showing a
constraint the database actually holds and one showing a guess."
```

---

### Task 12: `PostgresAdapter`

**Files:**
- Create: `engine/src/lumen/architect/adapters/postgres.py`
- Modify: `engine/src/lumen/architect/adapters/__init__.py` (export it)
- Modify: `engine/pyproject.toml` (add `asyncpg`)
- Test: `services/api/tests/test_postgres_adapter.py` (integration — needs a live instance)

**Interfaces:**
- Consumes: `DiscoveredColumn`, `DiscoveredTable`, `DiscoveredStructure` from Task 11.
- Produces: `PostgresAdapter(dsn: str, schema: str = "public")`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_postgres_adapter.py`:

```python
"""PostgresAdapter against a real database.

Uses the tenant instance as a stand-in for a customer's database — the
adapter cannot tell the difference, which is the point.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen.architect.adapters.postgres import PostgresAdapter
from lumen.architect.spec import SqlType
from lumen_api.settings import get_settings
from lumen_api.tenant_db import get_tenant_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db,
        reason="TENANT_DATABASE_URL is not configured",
    ),
]


@pytest.fixture
async def sample_schema():
    name = f"adapter_test_{uuid.uuid4().hex[:8]}"
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{name}"'))
        await conn.execute(
            text(
                f'CREATE TABLE "{name}".customers ('
                "  id text PRIMARY KEY,"
                "  name varchar(120) NOT NULL,"
                "  balance numeric(12,2)"
                ")"
            )
        )
        await conn.execute(
            text(
                f'CREATE TABLE "{name}".orders ('
                "  id bigint PRIMARY KEY,"
                f'  customer_id text REFERENCES "{name}".customers (id)'
                ")"
            )
        )
        await conn.execute(text(f"INSERT INTO \"{name}\".customers VALUES ('c1', 'Acme', 10.00)"))
    try:
        yield name
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


def _dsn() -> str:
    return get_settings().tenant_database_url.get_secret_value()


async def test_discovery_is_marked_declared(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    assert structure.declared is True


async def test_every_table_is_found(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    assert {t.name for t in structure.tables} == {"customers", "orders"}


async def test_the_real_primary_key_is_read_not_inferred(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    customers = next(t for t in structure.tables if t.name == "customers")
    assert customers.primary_key == ("id",)


async def test_the_real_foreign_key_is_read(sample_schema):
    """This is the whole argument for D10: when the source is a database,
    the relationship is read, not guessed at 95% containment."""
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    orders = next(t for t in structure.tables if t.name == "orders")
    assert orders.foreign_keys == (("customer_id", "customers", "id"),)


async def test_types_map_onto_the_closed_enum(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    customers = next(t for t in structure.tables if t.name == "customers")
    types = {c.name: (c.sql_type, c.type_arg) for c in customers.columns}
    assert types["id"] == (SqlType.TEXT, None)
    assert types["name"] == (SqlType.VARCHAR, "120")
    assert types["balance"] == (SqlType.NUMERIC, "12,2")


async def test_nullability_is_read(sample_schema):
    structure = await PostgresAdapter(_dsn(), sample_schema).discover()
    customers = next(t for t in structure.tables if t.name == "customers")
    assert next(c for c in customers.columns if c.name == "name").nullable is False


async def test_read_returns_rows(sample_schema):
    frame = await PostgresAdapter(_dsn(), sample_schema).read("customers")
    assert frame.height == 1


async def test_reading_an_undiscovered_table_is_refused(sample_schema):
    """A table name cannot be parameterised, so it is interpolated — which
    makes validating it against the discovered list the only thing standing
    between this and injection."""
    adapter = PostgresAdapter(_dsn(), sample_schema)
    await adapter.discover()
    with pytest.raises(ValueError, match="not discovered"):
        await adapter.read('customers"; DROP TABLE customers; --')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q -m integration tests/test_postgres_adapter.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen.architect.adapters.postgres'`

- [ ] **Step 3: Write the implementation**

Add `"asyncpg>=0.30"` to `dependencies` in `engine/pyproject.toml`, then `cd engine && uv sync`.

Create `engine/src/lumen/architect/adapters/postgres.py`:

```python
"""A live customer Postgres.

The reason D10 mirrors structure before copying a byte: this adapter does
not infer anything. Types, primary keys and foreign keys are read from the
customer's own information_schema, so the diagram shows constraints the
database actually holds rather than relationships guessed from containment.
"""

from __future__ import annotations

from typing import Any

import asyncpg
import polars as pl

from lumen.architect.adapters.base import DiscoveredColumn, DiscoveredStructure, DiscoveredTable
from lumen.architect.spec import SqlType

# Postgres's own type names onto our closed enum. Anything absent falls to
# TEXT, which is lossless for display and honest about what we understand.
_TYPES: dict[str, SqlType] = {
    "smallint": SqlType.INTEGER,
    "integer": SqlType.INTEGER,
    "bigint": SqlType.BIGINT,
    "numeric": SqlType.NUMERIC,
    "real": SqlType.DOUBLE,
    "double precision": SqlType.DOUBLE,
    "boolean": SqlType.BOOLEAN,
    "text": SqlType.TEXT,
    "character varying": SqlType.VARCHAR,
    "character": SqlType.VARCHAR,
    "date": SqlType.DATE,
    "timestamp without time zone": SqlType.TIMESTAMP,
    "timestamp with time zone": SqlType.TIMESTAMPTZ,
    "uuid": SqlType.UUID,
    "json": SqlType.JSONB,
    "jsonb": SqlType.JSONB,
}

_COLUMNS = """
select table_name, column_name, data_type, is_nullable,
       character_maximum_length, numeric_precision, numeric_scale
from information_schema.columns
where table_schema = $1
order by table_name, ordinal_position
"""

_PRIMARY_KEYS = """
select tc.table_name, kcu.column_name
from information_schema.table_constraints tc
join information_schema.key_column_usage kcu
  on kcu.constraint_name = tc.constraint_name
 and kcu.table_schema = tc.table_schema
where tc.table_schema = $1 and tc.constraint_type = 'PRIMARY KEY'
order by kcu.ordinal_position
"""

_FOREIGN_KEYS = """
select tc.table_name        as from_table,
       kcu.column_name      as from_column,
       ccu.table_name       as to_table,
       ccu.column_name      as to_column
from information_schema.table_constraints tc
join information_schema.key_column_usage kcu
  on kcu.constraint_name = tc.constraint_name
 and kcu.table_schema = tc.table_schema
join information_schema.constraint_column_usage ccu
  on ccu.constraint_name = tc.constraint_name
 and ccu.table_schema = tc.table_schema
where tc.table_schema = $1 and tc.constraint_type = 'FOREIGN KEY'
"""


def _type_of(row: Any) -> tuple[SqlType, str | None]:
    sql_type = _TYPES.get(row["data_type"], SqlType.TEXT)
    if sql_type is SqlType.VARCHAR and row["character_maximum_length"]:
        return sql_type, str(row["character_maximum_length"])
    if sql_type is SqlType.NUMERIC and row["numeric_precision"]:
        return sql_type, f"{row['numeric_precision']},{row['numeric_scale'] or 0}"
    return sql_type, None


class PostgresAdapter:
    kind = "postgres"
    supports_incremental = True

    def __init__(self, dsn: str, schema: str = "public") -> None:
        self._dsn = dsn
        self._schema = schema
        self._known: set[str] = set()

    async def discover(self) -> DiscoveredStructure:
        conn = await asyncpg.connect(self._dsn)
        try:
            columns = await conn.fetch(_COLUMNS, self._schema)
            primary = await conn.fetch(_PRIMARY_KEYS, self._schema)
            foreign = await conn.fetch(_FOREIGN_KEYS, self._schema)
        finally:
            await conn.close()

        by_table: dict[str, list[DiscoveredColumn]] = {}
        for row in columns:
            sql_type, type_arg = _type_of(row)
            by_table.setdefault(row["table_name"], []).append(
                DiscoveredColumn(
                    name=row["column_name"],
                    sql_type=sql_type,
                    type_arg=type_arg,
                    nullable=row["is_nullable"] == "YES",
                )
            )

        keys: dict[str, list[str]] = {}
        for row in primary:
            keys.setdefault(row["table_name"], []).append(row["column_name"])

        references: dict[str, list[tuple[str, str, str]]] = {}
        for row in foreign:
            references.setdefault(row["from_table"], []).append(
                (row["from_column"], row["to_table"], row["to_column"])
            )

        tables = tuple(
            DiscoveredTable(
                name=name,
                columns=tuple(cols),
                primary_key=tuple(keys[name]) if name in keys else None,
                foreign_keys=tuple(references.get(name, ())),
            )
            for name, cols in sorted(by_table.items())
        )
        self._known = {t.name for t in tables}
        return DiscoveredStructure(tables=tables, declared=True)

    async def read(self, table: str, limit: int | None = None) -> Any:
        # A table name cannot be a bind parameter, so it is interpolated —
        # which makes this check the only thing between the adapter and
        # injection. The name must be one discovery already returned; it is
        # never taken from user or model input.
        if not self._known:
            await self.discover()
        if table not in self._known:
            raise ValueError(f"table {table!r} was not discovered on this source")

        suffix = f" LIMIT {int(limit)}" if limit is not None else ""
        query = f'SELECT * FROM "{self._schema}"."{table}"{suffix}'
        conn = await asyncpg.connect(self._dsn)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
        return pl.DataFrame([dict(row) for row in rows])
```

Add to `engine/src/lumen/architect/adapters/__init__.py`:

```python
from lumen.architect.adapters.postgres import PostgresAdapter
```

and add `"PostgresAdapter"` to its `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q -m integration tests/test_postgres_adapter.py`
Expected: PASS — 8 passed

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/adapters/postgres.py engine/src/lumen/architect/adapters/__init__.py engine/pyproject.toml services/api/tests/test_postgres_adapter.py
git commit -m "feat(architect): live Postgres source adapter

Infers nothing. Types, primary keys and foreign keys all come from the
customer's own information_schema, which is the whole argument for D10:
when the source is a database, a relationship is read rather than guessed
at 95% containment, and the diagram shows what the database actually holds.

read() interpolates the table name because a table name cannot be a bind
parameter. Validating it against the discovered set is therefore the only
thing standing between this and injection, so it has its own test that
tries a name with a quote and a DROP in it."
```

---

### Task 13: `MySQLAdapter`

**Files:**
- Create: `engine/src/lumen/architect/adapters/mysql.py`
- Modify: `engine/src/lumen/architect/adapters/__init__.py` (export it)
- Modify: `engine/pyproject.toml` (add `aiomysql`)
- Test: `engine/tests/test_mysql_adapter.py`

**Note on testing:** there is no live MySQL in this project's environment, so this adapter is unit-tested against recorded `information_schema` rows rather than a live server. Do not write an integration test that cannot run — a skipped test that has never passed is worse than an honest unit test, because it looks like coverage.

**Interfaces:**
- Consumes: `DiscoveredColumn`, `DiscoveredTable`, `DiscoveredStructure` from Task 11.
- Produces: `MySQLAdapter(dsn: str, database: str)`, `_map_mysql_type(data_type: str, column_type: str, max_length: int | None, precision: int | None, scale: int | None) -> tuple[SqlType, str | None]`, `_build_structure(columns, keys) -> DiscoveredStructure`.

- [ ] **Step 1: Write the failing test**

Create `engine/tests/test_mysql_adapter.py`:

```python
"""MySQLAdapter's pure parts, against recorded information_schema rows.

No live MySQL exists in this environment. Testing the mapping and assembly
directly is honest; a skipped integration test that has never run would
look like coverage without being any.
"""

from __future__ import annotations

from lumen.architect.adapters.mysql import _build_structure, _map_mysql_type
from lumen.architect.spec import SqlType


def test_tinyint_one_is_boolean():
    """MySQL has no bool. tinyint(1) is the convention every ORM emits, and
    reading it as an integer would show a customer 0/1 where they wrote
    true/false."""
    assert _map_mysql_type("tinyint", "tinyint(1)", None, None, None)[0] is SqlType.BOOLEAN


def test_a_wider_tinyint_stays_an_integer():
    assert _map_mysql_type("tinyint", "tinyint(4)", None, None, None)[0] is SqlType.INTEGER


def test_int_and_bigint():
    assert _map_mysql_type("int", "int(11)", None, None, None)[0] is SqlType.INTEGER
    assert _map_mysql_type("bigint", "bigint(20)", None, None, None)[0] is SqlType.BIGINT


def test_varchar_carries_its_length():
    assert _map_mysql_type("varchar", "varchar(120)", 120, None, None) == (SqlType.VARCHAR, "120")


def test_decimal_carries_precision_and_scale():
    assert _map_mysql_type("decimal", "decimal(12,2)", None, 12, 2) == (SqlType.NUMERIC, "12,2")


def test_datetime_is_naive_and_timestamp_is_aware():
    """MySQL's timestamp converts to UTC on store; datetime does not. They
    are genuinely different types and collapsing them loses the zone."""
    assert _map_mysql_type("datetime", "datetime", None, None, None)[0] is SqlType.TIMESTAMP
    assert _map_mysql_type("timestamp", "timestamp", None, None, None)[0] is SqlType.TIMESTAMPTZ


def test_text_variants_all_map_to_text():
    for name in ("text", "mediumtext", "longtext", "tinytext"):
        assert _map_mysql_type(name, name, None, None, None)[0] is SqlType.TEXT


def test_json_maps_to_jsonb():
    assert _map_mysql_type("json", "json", None, None, None)[0] is SqlType.JSONB


def test_an_unknown_type_falls_back_to_text():
    assert _map_mysql_type("geometry", "geometry", None, None, None)[0] is SqlType.TEXT


def test_structure_assembly_reads_keys_from_key_column_usage():
    """MySQL puts REFERENCED_TABLE_NAME directly on KEY_COLUMN_USAGE, so
    unlike Postgres there is no referential_constraints join."""
    columns = [
        {"TABLE_NAME": "customers", "COLUMN_NAME": "id", "DATA_TYPE": "varchar",
         "COLUMN_TYPE": "varchar(36)", "IS_NULLABLE": "NO",
         "CHARACTER_MAXIMUM_LENGTH": 36, "NUMERIC_PRECISION": None, "NUMERIC_SCALE": None},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "id", "DATA_TYPE": "bigint",
         "COLUMN_TYPE": "bigint(20)", "IS_NULLABLE": "NO",
         "CHARACTER_MAXIMUM_LENGTH": None, "NUMERIC_PRECISION": 20, "NUMERIC_SCALE": 0},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "customer_id", "DATA_TYPE": "varchar",
         "COLUMN_TYPE": "varchar(36)", "IS_NULLABLE": "YES",
         "CHARACTER_MAXIMUM_LENGTH": 36, "NUMERIC_PRECISION": None, "NUMERIC_SCALE": None},
    ]
    keys = [
        {"TABLE_NAME": "customers", "COLUMN_NAME": "id", "CONSTRAINT_NAME": "PRIMARY",
         "REFERENCED_TABLE_NAME": None, "REFERENCED_COLUMN_NAME": None},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "id", "CONSTRAINT_NAME": "PRIMARY",
         "REFERENCED_TABLE_NAME": None, "REFERENCED_COLUMN_NAME": None},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "customer_id", "CONSTRAINT_NAME": "fk_cust",
         "REFERENCED_TABLE_NAME": "customers", "REFERENCED_COLUMN_NAME": "id"},
    ]

    structure = _build_structure(columns, keys)
    assert structure.declared is True

    customers = next(t for t in structure.tables if t.name == "customers")
    orders = next(t for t in structure.tables if t.name == "orders")
    assert customers.primary_key == ("id",)
    assert orders.primary_key == ("id",)
    assert orders.foreign_keys == (("customer_id", "customers", "id"),)
    assert next(c for c in orders.columns if c.name == "customer_id").nullable is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_mysql_adapter.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen.architect.adapters.mysql'`

- [ ] **Step 3: Write the implementation**

Add `"aiomysql>=0.2"` to `dependencies` in `engine/pyproject.toml`, then `cd engine && uv sync`.

Create `engine/src/lumen/architect/adapters/mysql.py`:

```python
"""A live customer MySQL.

Structurally the same job as the Postgres adapter, but MySQL's
information_schema differs in three ways that matter: KEY_COLUMN_USAGE
carries REFERENCED_TABLE_NAME directly so there is no referential_
constraints join, the "schema" is the database name, and the type
vocabulary is its own.
"""

from __future__ import annotations

from typing import Any

import aiomysql
import polars as pl

from lumen.architect.adapters.base import DiscoveredColumn, DiscoveredStructure, DiscoveredTable
from lumen.architect.spec import SqlType

_TYPES: dict[str, SqlType] = {
    "tinyint": SqlType.INTEGER,
    "smallint": SqlType.INTEGER,
    "mediumint": SqlType.INTEGER,
    "int": SqlType.INTEGER,
    "integer": SqlType.INTEGER,
    "bigint": SqlType.BIGINT,
    "decimal": SqlType.NUMERIC,
    "numeric": SqlType.NUMERIC,
    "float": SqlType.DOUBLE,
    "double": SqlType.DOUBLE,
    "bit": SqlType.BOOLEAN,
    "char": SqlType.VARCHAR,
    "varchar": SqlType.VARCHAR,
    "text": SqlType.TEXT,
    "tinytext": SqlType.TEXT,
    "mediumtext": SqlType.TEXT,
    "longtext": SqlType.TEXT,
    "date": SqlType.DATE,
    "datetime": SqlType.TIMESTAMP,
    # MySQL converts a timestamp to UTC on store and back on read; datetime
    # is stored verbatim. They are genuinely different and collapsing them
    # loses the zone.
    "timestamp": SqlType.TIMESTAMPTZ,
    "json": SqlType.JSONB,
}

_COLUMNS = """
select TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE,
       CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
from information_schema.COLUMNS
where TABLE_SCHEMA = %s
order by TABLE_NAME, ORDINAL_POSITION
"""

_KEYS = """
select TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME,
       REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
from information_schema.KEY_COLUMN_USAGE
where TABLE_SCHEMA = %s
order by TABLE_NAME, ORDINAL_POSITION
"""


def _map_mysql_type(
    data_type: str,
    column_type: str,
    max_length: int | None,
    precision: int | None,
    scale: int | None,
) -> tuple[SqlType, str | None]:
    """MySQL type to ours.

    tinyint(1) is special-cased because MySQL has no boolean and that is the
    spelling every ORM emits for one. Reading it as an integer would show a
    customer 0 and 1 where they wrote true and false.
    """
    normalised = data_type.strip().lower()

    if normalised == "tinyint" and column_type.replace(" ", "").lower().startswith("tinyint(1)"):
        return SqlType.BOOLEAN, None

    sql_type = _TYPES.get(normalised, SqlType.TEXT)

    if sql_type is SqlType.VARCHAR and max_length:
        return sql_type, str(max_length)
    if sql_type is SqlType.NUMERIC and precision:
        return sql_type, f"{precision},{scale or 0}"
    return sql_type, None


def _build_structure(
    columns: list[dict[str, Any]], keys: list[dict[str, Any]]
) -> DiscoveredStructure:
    by_table: dict[str, list[DiscoveredColumn]] = {}
    for row in columns:
        sql_type, type_arg = _map_mysql_type(
            row["DATA_TYPE"],
            row["COLUMN_TYPE"],
            row["CHARACTER_MAXIMUM_LENGTH"],
            row["NUMERIC_PRECISION"],
            row["NUMERIC_SCALE"],
        )
        by_table.setdefault(row["TABLE_NAME"], []).append(
            DiscoveredColumn(
                name=row["COLUMN_NAME"],
                sql_type=sql_type,
                type_arg=type_arg,
                nullable=row["IS_NULLABLE"] == "YES",
            )
        )

    primary: dict[str, list[str]] = {}
    references: dict[str, list[tuple[str, str, str]]] = {}
    for row in keys:
        table = row["TABLE_NAME"]
        if row["CONSTRAINT_NAME"] == "PRIMARY":
            primary.setdefault(table, []).append(row["COLUMN_NAME"])
        elif row["REFERENCED_TABLE_NAME"]:
            references.setdefault(table, []).append(
                (row["COLUMN_NAME"], row["REFERENCED_TABLE_NAME"], row["REFERENCED_COLUMN_NAME"])
            )

    tables = tuple(
        DiscoveredTable(
            name=name,
            columns=tuple(cols),
            primary_key=tuple(primary[name]) if name in primary else None,
            foreign_keys=tuple(references.get(name, ())),
        )
        for name, cols in sorted(by_table.items())
    )
    return DiscoveredStructure(tables=tables, declared=True)


class MySQLAdapter:
    kind = "mysql"
    supports_incremental = True

    def __init__(self, dsn: str, database: str) -> None:
        self._dsn = dsn
        self._database = database
        self._known: set[str] = set()

    async def _connect(self):
        from urllib.parse import urlparse

        parsed = urlparse(self._dsn)
        return await aiomysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "",
            password=parsed.password or "",
            db=self._database,
        )

    async def discover(self) -> DiscoveredStructure:
        conn = await self._connect()
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(_COLUMNS, (self._database,))
                columns = list(await cursor.fetchall())
                await cursor.execute(_KEYS, (self._database,))
                keys = list(await cursor.fetchall())
        finally:
            conn.close()

        structure = _build_structure(columns, keys)
        self._known = {t.name for t in structure.tables}
        return structure

    async def read(self, table: str, limit: int | None = None) -> Any:
        # Same reasoning as the Postgres adapter: a table name cannot be a
        # bind parameter, so validating against the discovered set is the
        # only barrier against injection.
        if not self._known:
            await self.discover()
        if table not in self._known:
            raise ValueError(f"table {table!r} was not discovered on this source")

        suffix = f" LIMIT {int(limit)}" if limit is not None else ""
        conn = await self._connect()
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(f"SELECT * FROM `{table}`{suffix}")
                rows = list(await cursor.fetchall())
        finally:
            conn.close()
        return pl.DataFrame(rows)
```

Add to `engine/src/lumen/architect/adapters/__init__.py`:

```python
from lumen.architect.adapters.mysql import MySQLAdapter
```

and add `"MySQLAdapter"` to its `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_mysql_adapter.py`
Expected: PASS — 12 passed

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/architect/adapters/mysql.py engine/src/lumen/architect/adapters/__init__.py engine/pyproject.toml engine/tests/test_mysql_adapter.py
git commit -m "feat(architect): live MySQL source adapter

Same job as the Postgres adapter against a different information_schema:
KEY_COLUMN_USAGE carries REFERENCED_TABLE_NAME directly so there is no
referential_constraints join, and the type vocabulary is its own.

tinyint(1) is special-cased to boolean because MySQL has no bool and that
is the spelling every ORM emits for one — reading it as an integer would
show a customer 0 and 1 where they wrote true and false. datetime and
timestamp are kept distinct because MySQL converts the latter to UTC on
store and collapsing them loses the zone.

Unit-tested against recorded information_schema rows rather than a live
server, because no MySQL exists in this environment and a skipped
integration test that has never run looks like coverage without being any."
```

---

### Task 14: The two-path read tier

**Files:**
- Create: `engine/src/lumen/datasets/sql_read.py`
- Modify: `engine/pyproject.toml` (add `duckdb`)
- Test: `services/api/tests/test_sql_read.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — takes a DSN, schema and table.
- Produces: `DUCKDB_ROW_THRESHOLD`, `read_table(dsn, schema, table, *, row_count=None, threshold=DUCKDB_ROW_THRESHOLD) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_sql_read.py`:

```python
"""The read tier. The parity test is the one that matters: a tiering that
changes results is a correctness bug, not a performance knob."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen.datasets.sql_read import read_table
from lumen_api.settings import get_settings
from lumen_api.tenant_db import get_tenant_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db,
        reason="TENANT_DATABASE_URL is not configured",
    ),
]


@pytest.fixture
async def sample_table():
    schema = f"read_test_{uuid.uuid4().hex[:8]}"
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.execute(
            text(f'CREATE TABLE "{schema}".rows (id bigint, label text, amount numeric(12,2))')
        )
        for i in range(50):
            await conn.execute(
                text(f'INSERT INTO "{schema}".rows VALUES (:i, :label, :amount)'),
                {"i": i, "label": f"row-{i}", "amount": i * 1.5},
            )
    try:
        yield schema
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


def _dsn() -> str:
    return get_settings().tenant_database_url.get_secret_value()


def test_the_polars_path_reads_every_row(sample_table):
    frame = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1_000_000)
    assert frame.height == 50


def test_the_duckdb_path_reads_every_row(sample_table):
    frame = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1)
    assert frame.height == 50


def test_both_paths_produce_identical_results(sample_table):
    """Forced by passing an explicit threshold rather than by generating
    five million rows."""
    low = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1_000_000)
    high = read_table(_dsn(), sample_table, "rows", row_count=50, threshold=1)

    assert low.columns == high.columns
    assert low.height == high.height
    assert low.sort("id").to_dicts() == high.sort("id").to_dicts()


def test_an_unknown_row_count_takes_the_polars_path(sample_table):
    """Unknown size is the common case on a first read. Defaulting to the
    simpler path means the accelerator is opt-in on evidence."""
    frame = read_table(_dsn(), sample_table, "rows", row_count=None)
    assert frame.height == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q -m integration tests/test_sql_read.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen.datasets.sql_read'`

- [ ] **Step 3: Write the implementation**

Add `"duckdb>=1.1"` to `dependencies` in `engine/pyproject.toml`, then `cd engine && uv sync`.

Create `engine/src/lumen/datasets/sql_read.py`:

```python
"""Reading a tenant table into a DataFrame.

Two paths, one contract. Below the threshold polars reads directly; above
it DuckDB's postgres extension streams and hands off Arrow, which becomes a
polars frame with no copy. Both return the same thing, which is why
`Backend` stays Literal["pandas", "polars", "spark"] — DuckDB is an
accelerator here, not a fourth backend, and adding it as one would touch
`validate_backend` plus every per-backend dispatch in materialize.py,
data_cleaning and statistics.

This reintroduces a threshold, which ADR-0013 spent its design removing.
The difference that makes it acceptable: ADR-0013's thresholds were
semantic — a wrong value produced wrong answers — while this one is a
compute choice with identical results either side, enforced by a parity
test. It is measured, not calibrated.
"""

from __future__ import annotations

from typing import Any

import polars as pl

# A conservative default. The real crossover is unmeasured and belongs in a
# benchmark, not a guess; until then the simpler path is the fallback.
DUCKDB_ROW_THRESHOLD = 5_000_000


def _read_with_polars(dsn: str, schema: str, table: str) -> pl.DataFrame:
    return pl.read_database_uri(query=f'SELECT * FROM "{schema}"."{table}"', uri=dsn)


def _read_with_duckdb(dsn: str, schema: str, table: str) -> pl.DataFrame:
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute("INSTALL postgres")
        connection.execute("LOAD postgres")
        connection.execute("ATTACH ? AS remote (TYPE postgres, READ_ONLY)", [dsn])
        # Identifiers cannot be bound, but both arrive from the tenant
        # naming functions and a validated SchemaSpec, never from input.
        return connection.execute(f'SELECT * FROM remote."{schema}"."{table}"').pl()
    finally:
        connection.close()


def read_table(
    dsn: str,
    schema: str,
    table: str,
    *,
    row_count: int | None = None,
    threshold: int = DUCKDB_ROW_THRESHOLD,
) -> Any:
    """The tenant table as a polars DataFrame.

    An unknown `row_count` takes the polars path: unknown size is the common
    case on a first read, and defaulting to the simpler path keeps the
    accelerator opt-in on evidence rather than on a guess.
    """
    if row_count is not None and row_count >= threshold:
        return _read_with_duckdb(dsn, schema, table)
    return _read_with_polars(dsn, schema, table)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q -m integration tests/test_sql_read.py`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add engine/src/lumen/datasets/sql_read.py engine/pyproject.toml services/api/tests/test_sql_read.py
git commit -m "feat(datasets): two-path read tier, polars and DuckDB

DuckDB is an accelerator, not a fourth backend (D12). Backend stays
Literal[pandas, polars, spark]; adding a value would touch validate_backend
plus every per-backend dispatch in materialize.py, data_cleaning and
statistics. The postgres extension hands off Arrow, so the polars frame is
zero-copy and D3's promise that the engine does not change survives.

The parity test carries this task. A tiering that changes results is a
correctness bug rather than a performance knob, and it is forced by passing
an explicit threshold rather than by generating five million rows.

An unknown row count takes the polars path, so the accelerator is opt-in on
evidence rather than on a guess."
```

- [ ] **Step 6: Run everything built so far**

Run: `cd engine && uv run pytest -q` then `cd services/api && uv run pytest -q -m "not integration"`
Expected: PASS on both.

---

# Phase 4 — Orchestration

### Task 15: `design_schema()`

**Files:**
- Create: `services/api/src/lumen_api/architect.py`
- Test: `services/api/tests/test_architect_design.py`

**Interfaces:**
- Consumes: `sanitize_identifier` (Task 2), `infer_sql_type` / `select_primary_key` / `detect_foreign_keys` (Tasks 3–5), `SchemaSpec` (Task 1), `FileAdapter` (Task 11).
- Produces: `design_schema(org_id, user_id, source_id) -> SchemaSpec`, `_semantic_pairs(db, org_id) -> list[tuple[str, str, str, str]]`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_architect_design.py`:

```python
"""design_schema against the live instances."""

from __future__ import annotations

import uuid

import httpx
import polars as pl
import pytest
import pytest_asyncio
from sqlalchemy import text

from lumen_api.architect import design_schema
from lumen_api.auth.dependencies import Identity
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import ensure_tenant_schema, tenant_raw_schema_name, tenant_session

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


def _admin_headers() -> dict[str, str]:
    key = get_settings().supabase_service_role_key.get_secret_value()
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _create_user() -> uuid.UUID:
    settings = get_settings()
    response = httpx.post(
        f"{settings.supabase_url}/auth/v1/admin/users",
        headers=_admin_headers(),
        json={
            "email": f"lumen-arch-{uuid.uuid4().hex[:12]}@example.com",
            "password": uuid.uuid4().hex + "Aa1!",
            "email_confirm": True,
            "user_metadata": {"display_name": "Architect Tester", "org_name": "Architect Org"},
        },
        timeout=30,
    )
    response.raise_for_status()
    return uuid.UUID(response.json()["id"])


def _delete_user(user_id: uuid.UUID) -> None:
    settings = get_settings()
    httpx.delete(
        f"{settings.supabase_url}/auth/v1/admin/users/{user_id}",
        headers=_admin_headers(),
        timeout=30,
    )


async def _identity_of(user_id: uuid.UUID) -> Identity:
    async with user_session(user_id) as db:
        row = (await db.execute(text("select * from public.current_identity()"))).mappings().first()
    return Identity(
        user_id=row["user_id"], email=row["email"], display_name=row["display_name"],
        avatar_url=row["avatar_url"], org_id=row["org_id"], org_name=row["org_name"],
        org_slug=row["org_slug"], plan_code=row["plan_code"], role=str(row["role"]),
    )


@pytest_asyncio.fixture
async def identity():
    user_id = _create_user()
    try:
        yield await _identity_of(user_id)
    finally:
        _delete_user(user_id)


async def _stage(identity: Identity, table: str, frame: pl.DataFrame) -> uuid.UUID:
    """Put a frame in staging and register a data_sources row for it."""
    await ensure_tenant_schema(identity.org_id)
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status, table_name) "
                "values (:id, :org, :name, 'csv', 'idle', :table)"
            ),
            {"id": source_id, "org": identity.org_id, "name": f"{table}.csv", "table": table},
        )
    dsn = get_settings().tenant_database_url.get_secret_value()
    frame.write_database(
        table_name=f"{tenant_raw_schema_name(identity.org_id)}.{table}",
        connection=dsn,
        if_table_exists="replace",
    )
    return source_id


async def test_a_single_source_gets_typed_columns_and_a_primary_key(identity):
    source_id = await _stage(
        identity, "customers", pl.DataFrame({"id": ["c1", "c2"], "balance": [1.5, 2.5]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)

    table = next(t for t in spec.tables if t.name == "customers")
    assert table.primary_key == ("id",)
    assert table.pk_rationale
    assert {c.name for c in table.columns} == {"id", "balance"}


async def test_a_cross_source_relationship_is_detected(identity):
    await _stage(identity, "customers", pl.DataFrame({"id": ["c1", "c2"]}))
    orders_id = await _stage(
        identity, "orders", pl.DataFrame({"id": ["o1", "o2"], "customer_id": ["c1", "c2"]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, orders_id)

    keys = [k for k in spec.foreign_keys if k.from_table == "orders"]
    assert len(keys) == 1
    assert (keys[0].to_table, keys[0].to_column) == ("customers", "id")
    assert keys[0].enforced is True


async def test_column_names_are_sanitised(identity):
    source_id = await _stage(
        identity, "weird", pl.DataFrame({"Customer ID": ["a"], "2024 Revenue": [1.0]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    names = {c.name for c in spec.tables[0].columns}
    assert names == {"customer_id", "col_2024_revenue"}


async def test_the_returned_spec_validates(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    spec.validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q -m integration tests/test_architect_design.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen_api.architect'`

- [ ] **Step 3: Write the implementation**

Create `services/api/src/lumen_api/architect.py`:

```python
"""The Data Architect — design, propose, apply.

Deterministic by default. Types, keys and relationships come from
statistics the engine already computes; the model's only job is to make the
names readable and write the rationale prose (Task 16), and it is allowed to
fail without stopping anything. That split is the same one ADR-0008 states
for detection and ADR-0013 for calibration, and it is what keeps the
keyless MockProvider path working end to end.
"""

from __future__ import annotations

import uuid
from typing import Any

import polars as pl
from sqlalchemy import text

from lumen.architect.ddl import sanitize_identifier
from lumen.architect.infer import detect_foreign_keys, infer_sql_type, select_primary_key
from lumen.architect.spec import ColumnSpec, SchemaSpec, TableSpec
from lumen.datasets.materialize import frame_schema
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_raw_schema_name, tenant_schema_name

BACKEND = "polars"


async def _semantic_pairs(org_id: uuid.UUID, user_id: uuid.UUID) -> list[tuple[str, str, str, str]]:
    """ADR-0009's canonical entities, shaped as FK candidates.

    A canonical entity says "these columns across sources are the same
    business concept" — which is exactly the hint that one of them
    references the other. The engine never learns what a canonical entity
    is; this function is the whole translation.
    """
    async with user_session(user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select s.table_name as table_name, m.column_name as column_name "
                    "from public.canonical_entity_members m "
                    "join public.canonical_entities e on e.id = m.entity_id "
                    "join public.data_sources s on s.id = m.source_id "
                    "where e.org_id = :org and e.status = 'approved' "
                    "  and s.table_name is not null"
                ),
                {"org": org_id},
            )
        ).mappings().all()

    members = [(r["table_name"], r["column_name"]) for r in rows]
    pairs: list[tuple[str, str, str, str]] = []
    for child_table, child_column in members:
        for parent_table, parent_column in members:
            if child_table != parent_table:
                pairs.append((child_table, child_column, parent_table, parent_column))
    return pairs


async def _staged_tables(org_id: uuid.UUID) -> dict[str, pl.DataFrame]:
    """Every table currently in this org's staging schema.

    Read wholesale because FK detection is inherently cross-table — a
    relationship cannot be found by looking at one source in isolation.
    """
    dsn = get_settings().tenant_database_url.get_secret_value()
    raw = tenant_raw_schema_name(org_id)

    names = pl.read_database_uri(
        query=(
            "select table_name from information_schema.tables "
            f"where table_schema = '{raw}'"
        ),
        uri=dsn,
    )
    return {
        name: pl.read_database_uri(query=f'select * from "{raw}"."{name}"', uri=dsn)
        for name in names["table_name"].to_list()
    }


async def design_schema(
    org_id: uuid.UUID, user_id: uuid.UUID, source_id: uuid.UUID
) -> SchemaSpec:
    """Design the modelled schema for this org, including `source_id`.

    Returns the whole org's spec rather than one table's, because a foreign
    key is a statement about two tables and cannot be designed from one.
    """
    async with user_session(user_id) as db:
        sources = (
            await db.execute(
                text(
                    "select id, table_name from public.data_sources "
                    "where org_id = :org and table_name is not null"
                ),
                {"org": org_id},
            )
        ).mappings().all()
    source_of = {r["table_name"]: r["id"] for r in sources}

    frames = await _staged_tables(org_id)

    tables: list[TableSpec] = []
    taken_tables: set[str] = set()
    for raw_name, frame in sorted(frames.items()):
        table_name = sanitize_identifier(raw_name, taken=taken_tables)
        taken_tables.add(table_name)

        taken_columns: set[str] = set()
        columns: list[ColumnSpec] = []
        for source_column, dtype in frame_schema(frame, BACKEND).items():
            name = sanitize_identifier(source_column, taken=taken_columns)
            taken_columns.add(name)
            sql_type, type_arg = infer_sql_type(dtype)
            columns.append(
                ColumnSpec(
                    name=name,
                    source_column=source_column,
                    sql_type=sql_type,
                    type_arg=type_arg,
                )
            )

        # PK selection reads the frame under its original column names; the
        # sanitised name is what goes into the spec.
        renamed = {c.source_column: c.name for c in columns}
        key, rationale = select_primary_key(frame, BACKEND, list(renamed))
        tables.append(
            TableSpec(
                name=table_name,
                source_id=source_of.get(raw_name, source_id),
                columns=tuple(columns),
                primary_key=tuple(renamed[k] for k in key) if key else None,
                pk_rationale=rationale,
                source_table=raw_name,
            )
        )

    table_tuple = tuple(tables)
    renamed_frames = {
        table.name: frames[table.source_table].rename(
            {c.source_column: c.name for c in table.columns}
        )
        for table in table_tuple
    }

    keys = detect_foreign_keys(
        renamed_frames,
        BACKEND,
        table_tuple,
        semantic_pairs=await _semantic_pairs(org_id, user_id),
    )

    spec = SchemaSpec(tables=table_tuple, foreign_keys=tuple(keys))
    spec.validate()
    return spec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q -m integration tests/test_architect_design.py`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/architect.py services/api/tests/test_architect_design.py
git commit -m "feat(architect): deterministic schema design

design_schema returns the whole org's spec rather than one table's, because
a foreign key is a statement about two tables and cannot be designed from
one in isolation.

_semantic_pairs is the entire translation between ADR-0009 and the engine.
A canonical entity says 'these columns across sources are the same business
concept', which is exactly the hint that one references the other — and the
engine never has to learn what a canonical entity is."
```

---

### Task 16: Model enrichment that is allowed to fail

**Files:**
- Modify: `services/api/src/lumen_api/architect.py` (append)
- Test: `services/api/tests/test_architect_enrichment.py`

**Interfaces:**
- Consumes: `provider()` from `lumen_api.llm`; `SchemaSpec` from Task 1.
- Produces: `enrich_spec(spec: SchemaSpec) -> SchemaSpec`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_architect_enrichment.py`:

```python
"""Enrichment must never be load-bearing.

The deterministic spec is the product; the model makes it readable. If the
provider is absent, denied or broken, the spec ships unchanged — which is
what keeps the keyless end-to-end path working.
"""

from __future__ import annotations

import uuid

import pytest

from lumen.architect.spec import ColumnSpec, SchemaSpec, SqlType, TableSpec
from lumen_api.architect import enrich_spec

_SRC = uuid.uuid4()


def _spec() -> SchemaSpec:
    return SchemaSpec(
        tables=(
            TableSpec(
                name="orders",
                source_id=_SRC,
                columns=(
                    ColumnSpec(name="id", source_column="id", sql_type=SqlType.TEXT),
                ),
                primary_key=("id",),
                pk_rationale="unique and complete",
            ),
        )
    )


async def test_a_failing_provider_returns_the_spec_unchanged(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr("lumen_api.architect.provider", explode)
    assert await enrich_spec(_spec()) == _spec()


async def test_a_provider_returning_nonsense_returns_the_spec_unchanged(monkeypatch):
    class _Garbage:
        def complete(self, *args, **kwargs):
            class _R:
                text = "not json at all"
            return _R()

    monkeypatch.setattr("lumen_api.architect.provider", lambda *a, **k: _Garbage())
    assert await enrich_spec(_spec()) == _spec()


async def test_enrichment_never_changes_structure(monkeypatch):
    """A model may improve prose. It may not invent a column, drop one, or
    change a type — those are the deterministic layer's decisions."""
    class _Meddling:
        def complete(self, *args, **kwargs):
            class _R:
                text = '{"tables": {"orders": {"pk_rationale": "the order identifier"}}}'
            return _R()

    monkeypatch.setattr("lumen_api.architect.provider", lambda *a, **k: _Meddling())
    enriched = await enrich_spec(_spec())

    assert enriched.tables[0].pk_rationale == "the order identifier"
    assert enriched.tables[0].columns == _spec().tables[0].columns
    assert enriched.tables[0].name == "orders"
    assert enriched.tables[0].primary_key == ("id",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q tests/test_architect_enrichment.py`
Expected: FAIL with `ImportError: cannot import name 'enrich_spec'`

- [ ] **Step 3: Write the implementation**

Append to `services/api/src/lumen_api/architect.py`:

```python
_ENRICH_PROMPT = """\
You are given a database schema that was designed deterministically from a \
customer's data. Improve only the human-facing prose.

Return JSON of the form:
  {"tables": {"<table>": {"pk_rationale": "<one clear sentence>"}}}

Rules:
- Do not rename anything. Do not add or remove tables or columns.
- Do not mention types, thresholds, or percentages.
- Write for a business user reading a schema for the first time.

Schema:
"""


async def enrich_spec(spec: SchemaSpec) -> SchemaSpec:
    """Better prose from the model, or the spec exactly as it came in.

    Deliberately impossible to make load-bearing. Every failure mode —
    provider down, quota denied, MockProvider running, malformed JSON,
    a model trying to rename a table — returns the deterministic spec.
    That is what lets a deployment with no API key still produce a real
    database, and it is why the model is not on the critical path.
    """
    summary = {
        table.name: {
            "columns": [c.name for c in table.columns],
            "primary_key": list(table.primary_key or ()),
        }
        for table in spec.tables
    }

    try:
        response = provider(tier="fast").complete(
            [{"role": "user", "content": _ENRICH_PROMPT + json.dumps(summary, indent=2)}]
        )
        payload = json.loads(response.text)
        improvements = payload["tables"]
    except Exception:  # noqa: BLE001 — every failure is the same non-event
        return spec

    tables = tuple(
        replace(table, pk_rationale=str(improvements[table.name]["pk_rationale"]))
        if isinstance(improvements.get(table.name), dict)
        and improvements[table.name].get("pk_rationale")
        else table
        for table in spec.tables
    )
    # Structure is never taken from the model — only the prose field is
    # substituted, on tables that already existed.
    return replace(spec, tables=tables)
```

Extend that module's imports:

```python
import json
from dataclasses import replace

from lumen_api.llm import provider
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q tests/test_architect_enrichment.py`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/architect.py services/api/tests/test_architect_enrichment.py
git commit -m "feat(architect): model enrichment that cannot become load-bearing

Every failure mode returns the deterministic spec: provider down, quota
denied, MockProvider running, malformed JSON, or a model trying to rename a
table. Only the pk_rationale prose field is ever substituted, and only on
tables that already exist.

That is what lets a deployment with no API key still produce a real
database — the spec calls it out as the reason for choosing deterministic
inference over an LLM agent, since MockProvider dispatches on tool names
and a new registry would have fallen through to doing nothing."
```

---

### Task 17: Proposals and the trust integration

**Files:**
- Modify: `services/api/src/lumen_api/architect.py` (append `propose_schema`)
- Modify: `services/api/src/lumen_api/trust.py` (new `structural_shape` cases)
- Test: `services/api/tests/test_trust.py` (append)

**Interfaces:**
- Consumes: `SchemaSpec` (Task 1), `MigrationPlan` (Task 7).
- Produces: `propose_schema(org_id, user_id, source_id, spec, kind) -> uuid.UUID`; `structural_shape` handling for `"schema_design"` and `"schema_migration"`.

**Why this task is not optional:** without these cases, `structural_shape()` returns `"schema_design:unclassified"`, ADR-0011 can never accrue a streak for the pattern, and Task 22's auto-apply can never engage. The evolution feature is dead without it.

- [ ] **Step 1: Write the failing test**

Append to `services/api/tests/test_trust.py`:

```python
# ── ADR-0024 schema kinds ───────────────────────────────────────────────


class TestSchemaShapes:
    def test_a_single_table_design_is_named_by_its_count(self):
        spec = {"tables": [{"name": "orders"}]}
        assert structural_shape("schema_design", spec) == "schema_design:1_table"

    def test_several_tables_pluralise(self):
        spec = {"tables": [{"name": "orders"}, {"name": "customers"}]}
        assert structural_shape("schema_design", spec) == "schema_design:2_tables"

    def test_a_design_with_no_tables_is_its_own_shape(self):
        assert structural_shape("schema_design", {"tables": []}) == "schema_design:empty"

    def test_an_additive_migration(self):
        spec = {"steps": [{"kind": "add_column", "reversible": True}]}
        assert structural_shape("schema_migration", spec) == "schema_migration:additive"

    def test_a_widening_migration(self):
        spec = {"steps": [{"kind": "widen_type", "reversible": True}]}
        assert structural_shape("schema_migration", spec) == "schema_migration:type_widening"

    def test_any_irreversible_step_makes_the_whole_migration_destructive(self):
        """One narrowing step among ten additive ones is still destructive.
        Trust is granted per shape, so a shape that can hide an irreversible
        step would let one be auto-applied on the strength of the others."""
        spec = {
            "steps": [
                {"kind": "add_column", "reversible": True},
                {"kind": "narrow_type", "reversible": False},
            ]
        }
        assert structural_shape("schema_migration", spec) == "schema_migration:destructive"

    def test_a_mixed_reversible_migration_reports_the_widening(self):
        spec = {
            "steps": [
                {"kind": "add_column", "reversible": True},
                {"kind": "widen_type", "reversible": True},
            ]
        }
        assert structural_shape("schema_migration", spec) == "schema_migration:type_widening"

    def test_the_shape_never_contains_a_table_or_column_name(self):
        spec = {"tables": [{"name": "customer_ssns"}, {"name": "salaries"}]}
        shape = structural_shape("schema_design", spec)
        assert "customer_ssns" not in shape
        assert "salaries" not in shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q tests/test_trust.py -k Schema`
Expected: FAIL — shapes come back as `schema_design:unclassified`

- [ ] **Step 3: Write the implementation**

In `services/api/src/lumen_api/trust.py`, add to `structural_shape()` before its final fallback:

```python
    if kind == "schema_design":
        count = len(spec.get("tables") or [])
        if count == 0:
            return "schema_design:empty"
        return f"schema_design:{count}_table" + ("s" if count != 1 else "")
    if kind == "schema_migration":
        steps = spec.get("steps") or []
        if not steps:
            return "schema_migration:empty"
        # Any irreversible step makes the whole migration destructive. Trust
        # is granted per shape, so a shape that could hide an irreversible
        # step would let one ride in on the strength of the reversible ones.
        if any(not step.get("reversible", False) for step in steps):
            return "schema_migration:destructive"
        if any(step.get("kind") == "widen_type" for step in steps):
            return "schema_migration:type_widening"
        return "schema_migration:additive"
```

Append to `services/api/src/lumen_api/architect.py`:

```python
async def propose_schema(
    org_id: uuid.UUID,
    user_id: uuid.UUID,
    source_id: uuid.UUID,
    spec: SchemaSpec,
    kind: str = "schema_design",
) -> uuid.UUID:
    """Write the spec as a Proposal for a human to accept.

    Non-negotiable #2: every mutating agent action produces one of these.
    Creating a customer's database is unambiguously mutating.
    """
    spec.validate()

    async with user_session(user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs "
                    "(org_id, source_id, thread_id, kind, status, backend, created_by) "
                    "values (:org, :source, gen_random_uuid(), 'architect', 'succeeded', "
                    "        'polars', :user) returning id"
                ),
                {"org": org_id, "source": source_id, "user": user_id},
            )
        ).scalar_one()

        proposal_id = (
            await db.execute(
                text(
                    "insert into public.proposals "
                    "(org_id, run_id, thread_id, author_agent, kind, spec, rationale) "
                    "values (:org, :run, :run, 'architect', :kind, cast(:spec as jsonb), "
                    "        :rationale) returning id"
                ),
                {
                    "org": org_id,
                    "run": run_id,
                    "kind": kind,
                    "spec": json.dumps(_spec_to_json(spec)),
                    "rationale": _describe(spec),
                },
            )
        ).scalar_one()
    return proposal_id


def _spec_to_json(spec: SchemaSpec) -> dict[str, Any]:
    return {
        "layout": spec.layout,
        "tables": [
            {
                "name": t.name,
                "source_id": str(t.source_id),
                "source_table": t.source_table,
                "primary_key": list(t.primary_key or ()),
                "pk_rationale": t.pk_rationale,
                "columns": [
                    {
                        "name": c.name,
                        "source_column": c.source_column,
                        "sql_type": c.sql_type.value,
                        "type_arg": c.type_arg,
                        "nullable": c.nullable,
                        "deprecated": c.deprecated,
                    }
                    for c in t.columns
                ],
            }
            for t in spec.tables
        ],
        "foreign_keys": [
            {
                "from_table": k.from_table,
                "from_column": k.from_column,
                "to_table": k.to_table,
                "to_column": k.to_column,
                "containment": k.containment,
                "enforced": k.enforced,
                "evidence": [e.value for e in k.evidence],
                "rationale": k.rationale,
            }
            for k in spec.foreign_keys
        ],
    }


def _describe(spec: SchemaSpec) -> str:
    enforced = sum(1 for k in spec.foreign_keys if k.enforced)
    observed = len(spec.foreign_keys) - enforced
    parts = [f"{len(spec.tables)} table{'s' if len(spec.tables) != 1 else ''}"]
    if enforced:
        parts.append(f"{enforced} enforced relationship{'s' if enforced != 1 else ''}")
    if observed:
        parts.append(f"{observed} observed relationship{'s' if observed != 1 else ''}")
    return "A database with " + ", ".join(parts) + "."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q tests/test_trust.py`
Expected: PASS — the existing trust tests plus 8 new ones

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/trust.py services/api/src/lumen_api/architect.py services/api/tests/test_trust.py
git commit -m "feat(architect): schema proposals and their trust signatures

structural_shape learns schema_design and schema_migration. Without these
cases the shape is 'unclassified', ADR-0011 can never accrue a streak for
the pattern, and the auto-apply in the evolution task can never engage —
this is a required integration, not a nice-to-have.

Any irreversible step makes a whole migration destructive rather than
averaging out. Trust is granted per shape, so a shape that could hide an
irreversible step would let one ride in on the strength of the reversible
ones around it."
```

---

### Task 18: Applying a schema

**Files:**
- Modify: `services/api/src/lumen_api/architect.py` (append `apply_schema`, `_spec_from_json`)
- Modify: `services/api/src/lumen_api/proposals.py` (dispatch)
- Test: `services/api/tests/test_architect_apply.py`

**Interfaces:**
- Consumes: `render_ddl` (Task 6), `tenant_session` (Task 8), `_spec_to_json` (Task 17).
- Produces: `apply_schema(org_id, user_id, spec) -> dict[str, Any]`, `_spec_from_json(payload) -> SchemaSpec`.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_architect_apply.py` (reuse the `identity`, `_stage` and user helpers from `test_architect_design.py` verbatim — repeated here rather than imported so the file stands alone):

```python
"""Applying a schema: DDL runs, then the control-plane record."""

from __future__ import annotations

import uuid

import polars as pl
import pytest
from sqlalchemy import text

from lumen_api.architect import apply_schema, design_schema, propose_schema
from lumen_api.db.session import user_session
from lumen_api.proposals import DecisionRequest, decide_proposal
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_schema_name, tenant_session

# identity, _create_user, _delete_user, _identity_of and _stage are copied
# from tests/test_architect_design.py — see the note in that file.
from tests.test_architect_design import _stage, identity  # noqa: F401

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


async def _columns(org_id: uuid.UUID, table: str) -> dict[str, str]:
    async with tenant_session(org_id) as db:
        rows = (
            await db.execute(
                text(
                    "select column_name, data_type from information_schema.columns "
                    "where table_schema = :schema and table_name = :table"
                ),
                {"schema": tenant_schema_name(org_id), "table": table},
            )
        ).mappings().all()
    return {r["column_name"]: r["data_type"] for r in rows}


async def test_applying_creates_the_table_with_real_types(identity):
    source_id = await _stage(
        identity, "customers", pl.DataFrame({"id": ["c1"], "balance": [1.5]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    columns = await _columns(identity.org_id, "customers")
    assert columns["id"] == "text"
    assert columns["balance"] == "double precision"


async def test_the_primary_key_is_really_enforced(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1", "c2"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    with pytest.raises(Exception):
        async with tenant_session(identity.org_id) as db:
            await db.execute(text("insert into customers (id) values ('x'), ('x')"))


async def test_the_foreign_key_is_really_enforced(identity):
    await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    orders_id = await _stage(
        identity, "orders", pl.DataFrame({"id": ["o1"], "customer_id": ["c1"]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, orders_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    with pytest.raises(Exception) as caught:
        async with tenant_session(identity.org_id) as db:
            await db.execute(
                text("insert into orders (id, customer_id) values ('o9', 'ghost')")
            )
    assert "foreign key" in str(caught.value).lower()


async def test_accepting_the_proposal_applies_it(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    proposal_id = await propose_schema(identity.org_id, identity.user_id, source_id, spec)

    result = await decide_proposal(proposal_id, DecisionRequest(decision="accept"), identity)
    assert result["status"] == "applied"
    assert await _columns(identity.org_id, "customers")


async def test_row_count_is_written(identity):
    """data_sources.row_count is SELECTed by three read paths and written by
    none — permanently NULL until now."""
    source_id = await _stage(
        identity, "customers", pl.DataFrame({"id": ["c1", "c2", "c3"]})
    )
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    async with user_session(identity.user_id) as db:
        count = (
            await db.execute(
                text("select row_count from public.data_sources where id = :id"),
                {"id": source_id},
            )
        ).scalar_one()
    assert count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q -m integration tests/test_architect_apply.py`
Expected: FAIL with `ImportError: cannot import name 'apply_schema'`

- [ ] **Step 3: Write the implementation**

Append to `services/api/src/lumen_api/architect.py`:

```python
def _spec_from_json(payload: dict[str, Any]) -> SchemaSpec:
    tables = tuple(
        TableSpec(
            name=t["name"],
            source_id=uuid.UUID(t["source_id"]),
            source_table=t.get("source_table"),
            primary_key=tuple(t["primary_key"]) or None,
            pk_rationale=t.get("pk_rationale", ""),
            columns=tuple(
                ColumnSpec(
                    name=c["name"],
                    source_column=c["source_column"],
                    sql_type=SqlType(c["sql_type"]),
                    type_arg=c.get("type_arg"),
                    nullable=c.get("nullable", True),
                    deprecated=c.get("deprecated", False),
                )
                for c in t["columns"]
            ),
        )
        for t in payload["tables"]
    )
    keys = tuple(
        ForeignKeySpec(
            from_table=k["from_table"],
            from_column=k["from_column"],
            to_table=k["to_table"],
            to_column=k["to_column"],
            containment=k["containment"],
            enforced=k["enforced"],
            evidence=tuple(Evidence(e) for e in k["evidence"]),
            rationale=k.get("rationale", ""),
        )
        for k in payload.get("foreign_keys", [])
    )
    return SchemaSpec(tables=tables, foreign_keys=keys, layout=payload.get("layout", "merged"))


async def apply_schema(
    org_id: uuid.UUID, user_id: uuid.UUID, spec: SchemaSpec
) -> dict[str, Any]:
    """Create the schema, load it from staging, then record it.

    Two sessions, in this order on purpose (spec §3.2). No transaction spans
    the instances, so one of them must go first — and writing the
    control-plane record LAST means a crash leaves an applied schema with a
    proposal still marked accepted, never a record of a schema that does not
    exist. Reconciliation is a re-run of discovery, which is authoritative.
    """
    await ensure_tenant_schema(org_id)
    schema = tenant_schema_name(org_id)
    raw = tenant_raw_schema_name(org_id)

    loaded: dict[str, int] = {}
    async with tenant_session(org_id) as db:
        for statement in render_ddl(spec, schema):
            await db.execute(text(statement))

        await db.execute(text("SET CONSTRAINTS ALL DEFERRED"))
        for table in spec.tables:
            columns = ", ".join(f'"{c.name}"' for c in table.columns)
            source = ", ".join(f'"{c.source_column}"' for c in table.columns)
            await db.execute(text(f'DELETE FROM "{schema}"."{table.name}"'))
            await db.execute(
                text(
                    f'INSERT INTO "{schema}"."{table.name}" ({columns}) '
                    f'SELECT {source} FROM "{raw}"."{table.source_table}"'
                )
            )
            loaded[table.name] = (
                await db.execute(text(f'SELECT count(*) FROM "{schema}"."{table.name}"'))
            ).scalar_one()

    async with user_session(user_id) as db:
        for table in spec.tables:
            await db.execute(
                text(
                    "update public.data_sources set table_name = :table, row_count = :rows "
                    "where id = :id"
                ),
                {"table": table.name, "rows": loaded[table.name], "id": table.source_id},
            )
            await record_artifact_dependency(
                db, org_id, "schema_table", table.name, table.source_id,
                [c.source_column for c in table.columns],
            )

    return {"tables": list(loaded), "rows": loaded, "schema": schema}
```

Extend the imports of that module:

```python
from lumen.architect.ddl import render_ddl, sanitize_identifier
from lumen.architect.spec import (
    ColumnSpec,
    Evidence,
    ForeignKeySpec,
    SchemaSpec,
    SqlType,
    TableSpec,
)
from lumen_api.lineage import record_artifact_dependency
from lumen_api.tenant_db import ensure_tenant_schema, tenant_raw_schema_name, tenant_schema_name, tenant_session
```

If `lumen_api/lineage.py` has no `record_artifact_dependency` with this signature, read that file and use the function it does expose for declaring an artifact's source columns — the ADR-0010 lineage row must be written here either way.

In `services/api/src/lumen_api/proposals.py`, add a branch to `decide_proposal`'s dispatch chain, immediately before the `_PIPELINE_KINDS` branch:

```python
    elif row["kind"] in ("schema_design", "schema_migration"):
        result = await _apply_schema_proposal(proposal_id, row, identity)
```

and add the handler alongside the other `_apply_*` functions:

```python
async def _apply_schema_proposal(
    proposal_id: uuid.UUID, row: Any, identity: Identity
) -> dict[str, Any]:
    """ADR-0024. The DDL runs on the tenant instance; the proposal's own
    status update is a control-plane write and therefore a second session."""
    from lumen_api.architect import _spec_from_json, apply_schema

    outcome = await apply_schema(
        identity.org_id, identity.user_id, _spec_from_json(dict(row["spec"] or {}))
    )

    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "update public.proposals set status = 'applied', decided_by = :user, "
                "       decided_at = now() where id = :id"
            ),
            {"user": identity.user_id, "id": proposal_id},
        )

    return {"id": str(proposal_id), "status": "applied", **outcome}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q -m integration tests/test_architect_apply.py`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/architect.py services/api/src/lumen_api/proposals.py services/api/tests/test_architect_apply.py
git commit -m "feat(architect): apply an accepted schema

Two sessions in a deliberate order. No transaction spans the instances, so
one must go first, and the control-plane record goes LAST: a crash then
leaves an applied schema with a proposal still marked accepted, never a
record of a schema that does not exist. Reconciliation is a re-run of
discovery, which is authoritative.

Also fixes a latent bug the spec names — data_sources.row_count is SELECTed
by three read paths and written by none, so it has been permanently NULL,
which is why the UI renders 'â€” rows'. Ingestion now knows the real count."
```

---

### Task 19: The staging load job

**Files:**
- Create: `services/worker/src/lumen_worker/ingest.py`
- Modify: `services/worker/src/lumen_worker/` worker settings (register the job — read how `process_schedule` and `diagnose_drift` are registered)
- Test: `services/worker/tests/test_ingest.py`

**Interfaces:**
- Consumes: `FileAdapter` (Task 11), `ensure_tenant_schema` / `tenant_raw_schema_name` (Tasks 8–9).
- Produces: `ingest_to_staging(ctx, source_id, org_id, acting_user_id) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

Create `services/worker/tests/test_ingest.py`:

```python
"""Staging load. Data must be visible before any schema is approved (D4)."""

from __future__ import annotations

import uuid

import polars as pl
import pytest
from sqlalchemy import text

from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_raw_schema_name, tenant_session
from lumen_worker.ingest import ingest_to_staging

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


class _FakeRedis:
    def __init__(self) -> None:
        self.jobs: list[tuple] = []

    async def enqueue_job(self, *args, **kwargs) -> None:
        self.jobs.append(args)


async def _seed(identity, name: str, content: bytes) -> uuid.UUID:
    path = f"org/{identity.org_id}/uploads/{uuid.uuid4().hex}{name[name.rfind('.'):]}"
    await SupabaseStorage().upload(path, content, "text/csv")
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, object_path) "
                "values (:id, :org, :name, 'csv', 'idle', :path)"
            ),
            {"id": source_id, "org": identity.org_id, "name": name, "path": path},
        )
    return source_id


async def test_a_csv_lands_in_staging(identity):
    source_id = await _seed(identity, "orders.csv", b"id,amount\n1,10\n2,20\n")
    ctx = {"redis": _FakeRedis()}

    result = await ingest_to_staging(
        ctx, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "staged"
    assert result["rows"] == 2

    async with tenant_session(identity.org_id) as db:
        count = (await db.execute(text("select count(*) from orders"))).scalar_one()
    assert count == 2


async def test_staging_enqueues_the_design_job(identity):
    source_id = await _seed(identity, "orders.csv", b"id\n1\n")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    assert any(job[0] == "design_schema_job" for job in ctx["redis"].jobs)


async def test_an_unsupported_format_marks_the_source_in_error(identity):
    source_id = await _seed(identity, "notes.docx", b"not a dataset")
    ctx = {"redis": _FakeRedis()}

    result = await ingest_to_staging(
        ctx, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "error"

    async with user_session(identity.user_id) as db:
        status = (
            await db.execute(
                text("select status from public.data_sources where id = :id"),
                {"id": source_id},
            )
        ).scalar_one()
    assert str(status) == "error"


async def test_an_unsupported_format_does_not_enqueue_a_design(identity):
    source_id = await _seed(identity, "notes.docx", b"nope")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    assert ctx["redis"].jobs == []


async def test_a_source_with_no_file_is_skipped(identity):
    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources (id, org_id, name, kind, status) "
                "values (:id, :org, 'empty.csv', 'csv', 'idle')"
            ),
            {"id": source_id, "org": identity.org_id},
        )
    result = await ingest_to_staging(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "skipped"
```

Add an `identity` fixture to this file by copying the `_create_user` / `_delete_user` / `_identity_of` helpers from `services/worker/tests/test_sentinel_diagnosis.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_ingest.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen_worker.ingest'`

- [ ] **Step 3: Write the implementation**

Create `services/worker/src/lumen_worker/ingest.py`:

```python
"""Ingestion jobs.

Two of them, and the split matters: staging lands data so a customer can
see it immediately (D4), and design proposes a schema. Keeping them apart
means a quota-denied or provider-down design never blocks a customer from
looking at the file they just uploaded.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any

from sqlalchemy import text

from lumen.architect.adapters.file import FileAdapter
from lumen.readers.exceptions import ReaderError
from lumen_api.datasets.store import SupabaseStorage
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import ensure_tenant_schema, tenant_raw_schema_name

SUPPORTED = (".csv", ".parquet", ".json", ".xlsx", ".xls")


async def _mark_error(user_id: uuid.UUID, source_id: uuid.UUID) -> None:
    async with user_session(user_id) as db:
        await db.execute(
            text("update public.data_sources set status = 'error' where id = :id"),
            {"id": source_id},
        )


async def ingest_to_staging(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Load a source into `tenant_<hex>_raw`, then enqueue its design.

    Staging is a permanent landing zone, not a temporary buffer: the
    raw-data browser reads it while a schema is awaiting review, and a
    failed promotion must be retryable without re-downloading the origin.
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    async with user_session(user_uuid) as db:
        source = (
            await db.execute(
                text("select name, object_path from public.data_sources where id = :id"),
                {"id": source_uuid},
            )
        ).mappings().first()

    if source is None or not source["object_path"]:
        return {"status": "skipped", "reason": "source has no uploaded file yet"}

    suffix = os.path.splitext(source["object_path"])[1].lower()
    if suffix not in SUPPORTED:
        await _mark_error(user_uuid, source_uuid)
        return {
            "status": "error",
            "reason": f"{suffix or 'this file'} is not a supported format; "
                      f"supported: {', '.join(SUPPORTED)}",
        }

    await ensure_tenant_schema(org_uuid)

    payload = await SupabaseStorage().download(source["object_path"])
    directory = tempfile.mkdtemp(prefix=f"lumen-ingest-{org_uuid.hex}-")
    local = os.path.join(directory, f"source{suffix}")
    with open(local, "wb") as file:
        file.write(payload)

    adapter = FileAdapter(local)
    try:
        frame = await adapter.read(adapter.table_name)
    except ReaderError:
        await _mark_error(user_uuid, source_uuid)
        return {"status": "error", "reason": "the file could not be read"}

    materialised = frame.collect() if hasattr(frame, "collect") else frame
    table = os.path.splitext(source["name"])[0]

    materialised.write_database(
        table_name=f"{tenant_raw_schema_name(org_uuid)}.{table}",
        connection=get_settings().tenant_database_url.get_secret_value(),
        if_table_exists="replace",
    )

    await ctx["redis"].enqueue_job("design_schema_job", source_id, org_id, acting_user_id)
    return {"status": "staged", "table": table, "rows": materialised.height}
```

Register `ingest_to_staging` in the worker's function list alongside `process_schedule` and `diagnose_drift`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_ingest.py`
Expected: PASS — 5 passed

- [ ] **Step 5: Commit**

```bash
git add services/worker/src/lumen_worker/ingest.py services/worker/tests/test_ingest.py
git commit -m "feat(worker): staging ingestion for any supported format

Staging is a permanent landing zone rather than a temporary buffer: the
raw-data browser reads it while a schema is awaiting review, and a failed
promotion must be retryable without re-downloading the origin.

Splitting staging from design is what stops a quota-denied or
provider-down design from blocking a customer looking at the file they just
uploaded — the data is visible either way, which is the whole point of D4."
```

---

### Task 20: The design job and the upload path

**Files:**
- Modify: `services/worker/src/lumen_worker/ingest.py` (append `design_schema_job`)
- Modify: `services/api/src/lumen_api/sources.py` (enqueue; stop assuming CSV)
- Test: `services/worker/tests/test_ingest.py` (append)

**Interfaces:**
- Consumes: `design_schema`, `enrich_spec`, `propose_schema` (Tasks 15–17).
- Produces: `design_schema_job(ctx, source_id, org_id, acting_user_id) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

Append to `services/worker/tests/test_ingest.py`:

```python
# ── the design job ──────────────────────────────────────────────────────

from lumen_worker.ingest import design_schema_job  # noqa: E402


async def test_the_design_job_creates_an_awaiting_review_proposal(identity):
    source_id = await _seed(identity, "customers.csv", b"id,name\nc1,Acme\nc2,Globex\n")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))

    result = await design_schema_job(
        ctx, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "proposed"

    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    "select kind, status from public.proposals where id = :id"
                ),
                {"id": uuid.UUID(result["proposal_id"])},
            )
        ).mappings().first()
    assert row["kind"] == "schema_design"
    assert str(row["status"]) == "awaiting_review"


async def test_nothing_is_created_in_the_modelled_schema_before_acceptance(identity):
    """D4: staging is immediate, promotion is approved. A table appearing
    before a human said yes would break the whole propose-then-apply spine."""
    source_id = await _seed(identity, "customers.csv", b"id\nc1\n")
    ctx = {"redis": _FakeRedis()}
    await ingest_to_staging(ctx, str(source_id), str(identity.org_id), str(identity.user_id))
    await design_schema_job(ctx, str(source_id), str(identity.org_id), str(identity.user_id))

    async with tenant_session(identity.org_id) as db:
        count = (
            await db.execute(
                text(
                    "select count(*) from information_schema.tables "
                    "where table_schema = :schema"
                ),
                {"schema": tenant_schema_name(identity.org_id)},
            )
        ).scalar_one()
    assert count == 0
```

Add `tenant_schema_name` to that file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_ingest.py -k design`
Expected: FAIL with `ImportError: cannot import name 'design_schema_job'`

- [ ] **Step 3: Write the implementation**

Append to `services/worker/src/lumen_worker/ingest.py`:

```python
async def design_schema_job(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Design a schema over everything in staging and propose it.

    Creates nothing in the modelled schema — that happens only when a human
    accepts the proposal (D4).
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    spec = await design_schema(org_uuid, user_uuid, source_uuid)
    spec = await enrich_spec(spec)
    proposal_id = await propose_schema(org_uuid, user_uuid, source_uuid, spec)

    return {"status": "proposed", "proposal_id": str(proposal_id), "tables": len(spec.tables)}
```

Extend that module's imports:

```python
from lumen_api.architect import design_schema, enrich_spec, propose_schema
```

Register `design_schema_job` in the worker's function list.

In `services/api/src/lumen_api/sources.py`, after the `data_sources` row is inserted in `upload_source`, enqueue the ingestion — and replace any hardcoded `.csv` in the object path with the uploaded file's real extension:

```python
    # The API never blocks on ingestion. The row exists, the file is in
    # Storage, and the customer sees the source immediately; staging and
    # design happen on the worker.
    await enqueue_job("ingest_to_staging", str(source_id), str(identity.org_id), str(identity.user_id))
```

Read that file for how it should reach redis — if it has no enqueue helper, add one that mirrors how the worker's `ctx["redis"]` is configured.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_ingest.py`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add services/worker/src/lumen_worker/ingest.py services/api/src/lumen_api/sources.py services/worker/tests/test_ingest.py
git commit -m "feat(architect): the design job, and wire ingestion into upload

Upload now enqueues staging, which enqueues design. The API blocks on
neither — the row exists and the file is in Storage, so the customer sees
their source immediately.

A test asserts the modelled schema is still empty after design runs. That
is D4's whole shape: staging is immediate, promotion is approved, and a
table appearing before a human said yes would break the propose-then-apply
spine every other agent in this product already follows.

Also stops the upload path assuming .csv — the object path now carries the
uploaded file's real extension, which is what makes the other four
supported formats reachable at all."
```

---

### Task 21: Re-ingest as full replacement

**Files:**
- Modify: `services/worker/src/lumen_worker/ingest.py` (append `refresh_source`)
- Test: `services/worker/tests/test_ingest_refresh.py`

**Interfaces:**
- Consumes: `render_replace` (Task 6), `tenant_session` (Task 8).
- Produces: `refresh_source(ctx, source_id, org_id, acting_user_id) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

Create `services/worker/tests/test_ingest_refresh.py`:

```python
"""Re-ingest. A file is a snapshot, so the table becomes exactly the file."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_session
from lumen_worker.ingest import refresh_source

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]

# identity, _FakeRedis, _seed and the applied-schema setup are copied from
# tests/test_ingest.py; see the note there.


async def test_a_replaced_snapshot_drops_rows_deleted_at_origin(identity, applied_customers):
    """An upsert would keep 'c2' forever. A file is a full snapshot, so a
    row the customer deleted at origin must disappear here too."""
    source_id, path = applied_customers  # seeded with c1,c2 and applied
    await _reupload(path, b"id\nc1\n")

    await refresh_source(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )

    async with tenant_session(identity.org_id) as db:
        ids = (await db.execute(text("select id from customers order by id"))).scalars().all()
    assert ids == ["c1"]


async def test_replacing_a_parent_does_not_violate_a_child_foreign_key(identity, applied_pair):
    """The reason foreign keys are DEFERRABLE and the replace runs inside
    one transaction: clearing customers while orders still references it
    would fail immediately otherwise."""
    customers_id, customers_path, _ = applied_pair
    await _reupload(customers_path, b"id\nc1\nc2\n")

    result = await refresh_source(
        {"redis": _FakeRedis()}, str(customers_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "refreshed"


async def test_a_violation_leaves_the_table_intact_and_records_drift(identity, applied_pair):
    """D5: an orphan row is a data-quality finding, not a crash. The
    transaction rolls back, so the table still holds what it held."""
    customers_id, customers_path, _ = applied_pair
    await _reupload(customers_path, b"id\nc9\n")  # orphans every order

    result = await refresh_source(
        {"redis": _FakeRedis()}, str(customers_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "constraint_violation"

    async with tenant_session(identity.org_id) as db:
        ids = (await db.execute(text("select id from customers order by id"))).scalars().all()
    assert "c1" in ids

    async with user_session(identity.user_id) as db:
        kinds = (
            await db.execute(
                text(
                    "select kind from public.drift_events "
                    "where source_id = :id order by occurred_at desc limit 1"
                ),
                {"id": customers_id},
            )
        ).scalars().all()
    assert kinds == ["schema_constraint"]
```

Build the `applied_customers` and `applied_pair` fixtures by composing Task 19's `_seed`, Task 19's `ingest_to_staging`, Task 15's `design_schema` and Task 18's `apply_schema`. `_reupload` uploads new bytes to the same `object_path` and waits for Storage convergence exactly as `services/worker/tests/test_sentinel_diagnosis.py::_upload_and_wait` does — copy that helper, it exists because Supabase Storage is not read-your-writes.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_ingest_refresh.py`
Expected: FAIL with `ImportError: cannot import name 'refresh_source'`

- [ ] **Step 3: Write the implementation**

Append to `services/worker/src/lumen_worker/ingest.py`:

```python
async def refresh_source(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Reload a source whose file changed, replacing the table's contents.

    Full replacement, not upsert: a file is a snapshot of the whole source,
    so a row the customer deleted at origin must disappear here. An upsert
    would keep it alive forever with nobody able to say why.

    Everything runs in one transaction with constraints deferred, so a
    parent can be cleared while a child still references it. A violation at
    COMMIT means the new snapshot genuinely breaks a relationship the data
    used to satisfy — that is a finding, not a crash (D5), so the
    transaction rolls back and a DriftEvent records it.
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    staged = await ingest_to_staging(ctx, source_id, org_id, acting_user_id)
    if staged["status"] != "staged":
        return staged

    async with user_session(user_uuid) as db:
        row = (
            await db.execute(
                text("select table_name from public.data_sources where id = :id"),
                {"id": source_uuid},
            )
        ).mappings().first()
    table = row["table_name"] if row else None
    if not table:
        # Never promoted — the data is in staging and that is correct.
        return {"status": "staged_only", "reason": "this source has no approved schema yet"}

    schema = tenant_schema_name(org_uuid)
    raw = tenant_raw_schema_name(org_uuid)

    try:
        async with tenant_session(org_uuid) as db:
            await db.execute(text("SET CONSTRAINTS ALL DEFERRED"))
            await db.execute(text(f'DELETE FROM "{schema}"."{table}"'))
            await db.execute(
                text(f'INSERT INTO "{schema}"."{table}" SELECT * FROM "{raw}"."{table}"')
            )
    except Exception as exc:  # noqa: BLE001 — see the docstring
        async with user_session(user_uuid) as db:
            await db.execute(
                text(
                    "insert into public.drift_events "
                    "(org_id, source_id, kind, severity, details) "
                    "values (:org, :source, 'schema_constraint', 0.8, cast(:details as jsonb))"
                ),
                {
                    "org": org_uuid,
                    "source": source_uuid,
                    "details": json.dumps({"error": str(exc), "table": table}),
                },
            )
        return {"status": "constraint_violation", "table": table}

    return {"status": "refreshed", "table": table}
```

Extend that module's imports with `json` and `tenant_schema_name`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_ingest_refresh.py`
Expected: PASS — 3 passed

- [ ] **Step 5: Commit**

```bash
git add services/worker/src/lumen_worker/ingest.py services/worker/tests/test_ingest_refresh.py
git commit -m "feat(worker): re-ingest as full snapshot replacement

A file is a snapshot of the whole source, so the table becomes exactly the
file. An upsert would keep a row the customer deleted at origin alive
forever with nobody able to say why.

One transaction with constraints deferred, which is why they were declared
DEFERRABLE: clearing a parent while a child still references it would fail
immediately otherwise. A violation at COMMIT means the new snapshot
genuinely breaks a relationship the data used to satisfy — the transaction
rolls back and a DriftEvent records it, because per D5 an orphan row is a
data-quality finding rather than a crash."
```

---

### Task 22: Schema evolution

**Files:**
- Modify: `services/worker/src/lumen_worker/ingest.py` (append `evolve_schema`)
- Test: `services/worker/tests/test_schema_evolution.py`

**Interfaces:**
- Consumes: `classify_migration` / `render_migration` (Task 7), `is_auto_apply_eligible` and `structural_shape` from `lumen_api.trust`, `propose_schema` (Task 17).
- Produces: `evolve_schema(ctx, source_id, org_id, acting_user_id) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

Create `services/worker/tests/test_schema_evolution.py`:

```python
"""Evolution. Reversible changes may earn autonomy; destructive ones never do."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_schema_name, tenant_session
from lumen_worker.ingest import evolve_schema

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


async def _earn_trust(identity, pattern: str, streak: int = 20) -> None:
    """Pre-seed what is_auto_apply_eligible requires. The real accept/reject
    cycle is exercised in services/api/tests/test_trust_learning.py; this
    file's subject is evolution, which needs trust as a precondition."""
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.pattern_trust_scores "
                "(org_id, pattern_signature, approvals, rejections, "
                " consecutive_approvals, score) "
                "values (:org, :pattern, :streak, 0, :streak, 0.9) "
                "on conflict (org_id, pattern_signature) do update set "
                "  consecutive_approvals = :streak, auto_apply_enabled = true"
            ),
            {"org": identity.org_id, "pattern": pattern, "streak": streak},
        )


async def _columns(org_id: uuid.UUID, table: str) -> set[str]:
    async with tenant_session(org_id) as db:
        rows = (
            await db.execute(
                text(
                    "select column_name from information_schema.columns "
                    "where table_schema = :schema and table_name = :table"
                ),
                {"schema": tenant_schema_name(org_id), "table": table},
            )
        ).scalars().all()
    return set(rows)


async def test_an_additive_change_auto_applies_when_trusted(identity, applied_customers):
    source_id, path = applied_customers
    await _earn_trust(identity, "schema_migration:additive")
    await _reupload(path, b"id,email\nc1,a@example.com\n")

    result = await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "applied"
    assert "email" in await _columns(identity.org_id, "customers")


async def test_the_same_change_awaits_review_when_not_trusted(identity, applied_customers):
    """The genuinely new behaviour: a reversible migration is eligible for
    autonomy, not entitled to it."""
    source_id, path = applied_customers
    await _reupload(path, b"id,email\nc1,a@example.com\n")

    result = await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "proposed"
    assert "email" not in await _columns(identity.org_id, "customers")


async def test_a_destructive_change_is_always_proposed_even_at_maximum_trust(
    identity, applied_typed
):
    """No trust level skips this. ADR-0017 §3 makes irreversibility a
    ceiling, and D7 keeps it there."""
    source_id, path = applied_typed  # a column currently typed text
    await _earn_trust(identity, "schema_migration:destructive", streak=500)
    await _reupload(path, b"id,amount\nc1,1\n")  # text -> integer, narrowing

    result = await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert result["status"] == "proposed"


async def test_a_column_absent_at_origin_is_kept(identity, applied_two_column):
    """D7 forbids DROP COLUMN outright. The column stays and is marked
    deprecated in the spec."""
    source_id, path = applied_two_column  # id, legacy
    await _earn_trust(identity, "schema_migration:additive")
    await _reupload(path, b"id\nc1\n")

    await evolve_schema(
        {"redis": _FakeRedis()}, str(source_id), str(identity.org_id), str(identity.user_id)
    )
    assert "legacy" in await _columns(identity.org_id, "customers")
```

Build the fixtures by composing Tasks 19, 15 and 18 exactly as Task 21 does; reuse the same `_reupload` and `_FakeRedis` helpers.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_schema_evolution.py`
Expected: FAIL with `ImportError: cannot import name 'evolve_schema'`

- [ ] **Step 3: Write the implementation**

Append to `services/worker/src/lumen_worker/ingest.py`:

```python
async def evolve_schema(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Compare a re-staged source against its applied schema and migrate.

    Two gates, and only one of them is negotiable. A reversible plan may
    auto-apply if this org has earned trust in that shape (ADR-0011); a
    plan with any irreversible step is always a proposal, at any trust
    level, because ADR-0017 §3 makes irreversibility a ceiling no learned
    signal may raise.
    """
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )

    staged = await ingest_to_staging(ctx, source_id, org_id, acting_user_id)
    if staged["status"] != "staged":
        return staged

    current = await current_spec(org_uuid, user_uuid)
    proposed = await design_schema(org_uuid, user_uuid, source_uuid)
    plan = classify_migration(current, proposed)

    if not plan.steps:
        return await refresh_source(ctx, source_id, org_id, acting_user_id)

    shape = structural_shape("schema_migration", {"steps": [vars(s) for s in plan.steps]})

    trusted = False
    if plan.reversible:
        async with user_session(user_uuid) as db:
            trusted = await is_auto_apply_eligible(db, org_uuid, shape)

    if not (plan.reversible and trusted):
        spec = await enrich_spec(proposed)
        proposal_id = await propose_schema(
            org_uuid, user_uuid, source_uuid, spec, kind="schema_migration"
        )
        return {"status": "proposed", "proposal_id": str(proposal_id), "shape": shape}

    async with tenant_session(org_uuid) as db:
        for statement in render_migration(plan, tenant_schema_name(org_uuid)):
            await db.execute(text(statement))

    await refresh_source(ctx, source_id, org_id, acting_user_id)
    return {"status": "applied", "shape": shape, "steps": len(plan.steps)}
```

Add `current_spec(org_id, user_id) -> SchemaSpec` to `services/api/src/lumen_api/architect.py`, reading the org's applied structure back out of `information_schema` on the tenant instance and assembling a `SchemaSpec` — the same assembly `PostgresAdapter.discover()` performs, pointed at our own tenant schema. Reuse `PostgresAdapter(dsn, tenant_schema_name(org_id))` rather than writing a second reader.

Extend `ingest.py`'s imports:

```python
from lumen.architect.migrate import classify_migration, render_migration
from lumen_api.architect import current_spec, design_schema, enrich_spec, propose_schema
from lumen_api.tenant_db import tenant_session
from lumen_api.trust import is_auto_apply_eligible, structural_shape
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/worker && uv run pytest -q -m integration tests/test_schema_evolution.py`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/worker/src/lumen_worker/ingest.py services/api/src/lumen_api/architect.py services/worker/tests/test_schema_evolution.py
git commit -m "feat(worker): schema evolution gated by reversibility and trust

Two gates, one of them non-negotiable. A reversible plan may auto-apply if
the org earned trust in that shape; a plan with any irreversible step is
always a proposal at any trust level, because ADR-0017 §3 makes
irreversibility a ceiling no learned signal may raise.

current_spec reads the applied structure back through PostgresAdapter
pointed at our own tenant schema, rather than a second information_schema
reader — the adapter already does exactly that job and a customer database
and our own are the same shape to it."
```

---

### Task 23: Switch the substrate and delete the orphans

**Files:**
- Modify: `services/api/src/lumen_api/datasets/store.py` (`resolve()`)
- Delete: `engine/src/lumen/database/postgres_manager.py`
- Delete: `engine/src/lumen/agents/postgres_admin_agent.py`
- Test: `services/api/tests/test_handle_store_substrate.py`

**Interfaces:**
- Consumes: `read_table` (Task 14), `tenant_schema_name` (Task 8).
- Produces: no new names — `HandleStore.resolve()` keeps its signature exactly, which is what leaves ADR-0008 through ADR-0013 untouched.

- [ ] **Step 1: Verify nothing imports the orphans**

Run:

```bash
cd "C:/Users/justi/Desktop/EDA-project" && grep -rn "postgres_manager\|PostgresAdminAgent\|PostgresManager" --include="*.py" engine/src services/api/src services/worker/src
```

Expected: matches only inside the two files being deleted. If anything else appears, stop and resolve it before continuing.

- [ ] **Step 2: Write the failing test**

Create `services/api/tests/test_handle_store_substrate.py`:

```python
"""resolve() reads SQL now. The contract is unchanged, which is the point:
ADR-0008 through ADR-0013 all consume a DataFrame and none of them change."""

from __future__ import annotations

import pytest

from lumen_api.settings import get_settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


async def test_resolve_returns_the_applied_table(identity, applied_customers):
    from lumen_api.datasets.store import HandleStore

    store = HandleStore(identity.org_id, identity.user_id)
    handle = await store.latest_for_source(applied_customers[0])
    frame = await store.resolve(handle.rid)

    materialised = frame.collect() if hasattr(frame, "collect") else frame
    assert materialised.height > 0


async def test_the_orphan_modules_are_gone():
    with pytest.raises(ModuleNotFoundError):
        __import__("lumen.database.postgres_manager")
    with pytest.raises(ModuleNotFoundError):
        __import__("lumen.agents.postgres_admin_agent")
```

- [ ] **Step 3: Make the change**

In `services/api/src/lumen_api/datasets/store.py`, change `resolve()` to read from the tenant schema through `read_table` instead of downloading Parquet, keeping its signature and return type identical. Read the current implementation first; the handle row already carries what is needed to find the table, and `data_sources.row_count` (now written, Task 18) supplies `row_count` for the tier decision.

Then delete both orphans:

```bash
git rm engine/src/lumen/database/postgres_manager.py engine/src/lumen/agents/postgres_admin_agent.py
```

- [ ] **Step 4: Run the full regression suite**

Run each, in order:

```bash
cd engine && uv run pytest -q
cd services/api && uv run pytest -q -m "not integration"
cd services/api && uv run pytest -q -m integration
cd services/worker && uv run pytest -q -m integration
```

Expected: PASS on all four. Note that `services/worker/tests/test_sentinel_diagnosis.py` has a known transient failure against Supabase Storage's eventual consistency under full-suite load — if exactly that test fails, re-run it in isolation to confirm before treating it as a regression.

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/datasets/store.py services/api/tests/test_handle_store_substrate.py engine/src/lumen/database/postgres_manager.py engine/src/lumen/agents/postgres_admin_agent.py
git commit -m "feat(architect): SQL becomes the substrate; delete the orphans

resolve() reads the tenant table instead of downloading Parquet, and its
signature does not change — which is exactly why ADR-0008 through ADR-0013
need no edits. Their contract was always 'give me a DataFrame' and it still
is; Parquet is now a compute cache rather than the source of truth (D3).

postgres_manager.py and postgres_admin_agent.py go. They were orphans from
the pre-SaaS codebase, imported by nothing, single-tenant, env-var driven —
and postgres_manager.execute_query(query: str) took whatever string it was
handed, which is precisely the shape this whole design exists to replace."
```

---

### Task 24: Multi-source layout and cross-source keys

Covers spec §3.6 and testing items 11 and 13, which Tasks 1–23 leave uncovered: `SchemaSpec.layout` exists but nothing sets it, and `sanitize_identifier`'s collision suffix produces `users_2` where §3.6 specifies `crm__users`.

**Files:**
- Modify: `engine/src/lumen/architect/ddl.py` (add `qualify_table_name`)
- Modify: `services/api/src/lumen_api/architect.py` (use it in `design_schema`)
- Test: `engine/tests/test_architect_ddl.py` (append)
- Test: `services/api/tests/test_cross_source_keys.py`

**Interfaces:**
- Consumes: `sanitize_identifier` (Task 2), `design_schema` (Task 15), `apply_schema` (Task 18).
- Produces: `qualify_table_name(table: str, alias: str | None, *, taken: set[str]) -> str`.

- [ ] **Step 1: Write the failing unit test**

Append to `engine/tests/test_architect_ddl.py`:

```python
# ── multi-source naming (§3.6) ──────────────────────────────────────────

from lumen.architect.ddl import qualify_table_name  # noqa: E402


def test_a_name_without_a_collision_keeps_its_own_name():
    """The customer uploaded orders.csv; they should see 'orders'. Prefixing
    everything defensively would make every table unrecognisable to prevent
    a collision that usually does not happen."""
    assert qualify_table_name("orders", "crm", taken=set()) == "orders"


def test_a_collision_takes_the_source_alias_as_a_prefix():
    assert qualify_table_name("users", "crm", taken={"users"}) == "crm__users"


def test_a_collision_with_no_alias_falls_back_to_a_numeric_suffix():
    assert qualify_table_name("users", None, taken={"users"}) == "users_2"


def test_a_prefixed_name_that_also_collides_gets_a_suffix():
    assert qualify_table_name("users", "crm", taken={"users", "crm__users"}) == "crm__users_2"


def test_the_alias_is_sanitised_too():
    assert qualify_table_name("users", "CRM Prod!", taken={"users"}) == "crm_prod__users"


def test_the_result_still_fits_the_identifier_limit():
    result = qualify_table_name("t" * 60, "a" * 60, taken={"t" * 60})
    assert len(result.encode("utf-8")) <= 63
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd engine && uv run pytest -q tests/test_architect_ddl.py -k qualify`
Expected: FAIL with `ImportError: cannot import name 'qualify_table_name'`

- [ ] **Step 3: Write the implementation**

Append to `engine/src/lumen/architect/ddl.py`:

```python
def qualify_table_name(table: str, alias: str | None, *, taken: set[str]) -> str:
    """A table's name in the merged schema, prefixed only on collision.

    §3.6's rule, and the reason it is "only on collision": the customer
    uploaded `orders.csv` and should see `orders`. Prefixing every table
    defensively would make all of them unrecognisable in order to prevent a
    collision that usually never happens.
    """
    base = sanitize_identifier(table)
    if base not in taken:
        return base

    if alias:
        # Built from the two already-sanitised halves rather than by
        # sanitising "alias__table" as one string: sanitize_identifier
        # collapses runs of underscores, which would turn the double
        # underscore into a single one and lose the visual distinction
        # between a prefix and an ordinary underscore in a table name.
        prefixed = _truncate_bytes(
            f"{sanitize_identifier(alias)}__{base}", MAX_IDENTIFIER_BYTES
        )
        if prefixed not in taken:
            return prefixed
        return sanitize_identifier(prefixed, taken=taken)

    return sanitize_identifier(table, taken=taken)
```

In `services/api/src/lumen_api/architect.py`, replace the table-naming line in `design_schema`:

```python
        table_name = sanitize_identifier(raw_name, taken=taken_tables)
```

with the alias-aware form, where the alias is the source's own name without its extension:

```python
        alias = alias_of.get(raw_name)
        table_name = qualify_table_name(raw_name, alias, taken=taken_tables)
```

and build `alias_of` alongside `source_of` from the same query, adding `name` to its SELECT:

```python
    alias_of = {
        r["table_name"]: os.path.splitext(r["name"])[0] for r in sources
    }
```

- [ ] **Step 4: Write the failing integration test**

Create `services/api/tests/test_cross_source_keys.py`:

```python
"""Testing items 11 and 13: a foreign key that crosses between two of the
customer's own databases, and the layout choice round-tripping.

This is the test that proves D11 is real rather than decorative. Postgres
enforces a foreign key across schemas within a database and never across
databases — putting every source of an org in one database is precisely
what makes this possible.
"""

from __future__ import annotations

import uuid

import polars as pl
import pytest
from sqlalchemy import text

from lumen_api.architect import apply_schema, design_schema
from lumen_api.settings import get_settings
from lumen_api.tenant_db import tenant_session

from tests.test_architect_design import _stage, identity  # noqa: F401

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


async def test_a_key_spans_two_separately_connected_sources(identity):
    """`customers` arrives from one source and `orders` from another. The
    relationship between them must be enforced, not merely drawn."""
    await _stage(identity, "customers", pl.DataFrame({"id": ["c1", "c2"]}))
    orders_id = await _stage(
        identity, "orders", pl.DataFrame({"id": ["o1"], "customer_id": ["c1"]})
    )

    spec = await design_schema(identity.org_id, identity.user_id, orders_id)
    await apply_schema(identity.org_id, identity.user_id, spec)

    key = next(k for k in spec.foreign_keys if k.from_table == "orders")
    assert key.to_table == "customers"
    assert key.enforced is True
    # The two tables came from two different data_sources rows.
    assert {t.source_id for t in spec.tables} != {orders_id}

    with pytest.raises(Exception) as caught:
        async with tenant_session(identity.org_id) as db:
            await db.execute(
                text("insert into orders (id, customer_id) values ('o2', 'ghost')")
            )
    assert "foreign key" in str(caught.value).lower()


async def test_two_sources_with_the_same_table_name_do_not_overwrite(identity):
    await _stage(identity, "users", pl.DataFrame({"id": ["a"]}))
    second = await _stage(identity, "users", pl.DataFrame({"id": ["b"], "extra": [1]}))

    spec = await design_schema(identity.org_id, identity.user_id, second)
    names = {t.name for t in spec.tables}
    assert len(names) == 2, f"one source overwrote the other: {names}"


async def test_the_layout_is_recorded_on_the_spec(identity):
    source_id = await _stage(identity, "customers", pl.DataFrame({"id": ["c1"]}))
    spec = await design_schema(identity.org_id, identity.user_id, source_id)
    assert spec.layout == "merged"
```

- [ ] **Step 5: Run both suites to verify they pass**

Run: `cd engine && uv run pytest -q tests/test_architect_ddl.py`
Run: `cd services/api && uv run pytest -q -m integration tests/test_cross_source_keys.py`
Expected: PASS on both.

- [ ] **Step 6: Commit**

```bash
git add engine/src/lumen/architect/ddl.py services/api/src/lumen_api/architect.py engine/tests/test_architect_ddl.py services/api/tests/test_cross_source_keys.py
git commit -m "feat(architect): source-aware table naming and cross-source keys

§3.6's rule: prefix on collision only. The customer uploaded orders.csv and
should see 'orders'; prefixing every table defensively would make all of
them unrecognisable to prevent a collision that usually never happens.

The integration test is what proves D11 is real rather than decorative.
Postgres enforces a foreign key across schemas within a database and never
across databases — putting every source of an org in one database is
exactly what makes 'the FK crosses between two of the customer's databases'
an enforced constraint instead of a line on a picture."
```

---

### Task 25: Connecting a customer database

Covers D10's second half — mirroring structure on connect, and copying only the tables the customer selects. Tasks 12 and 13 built the adapters; nothing yet reaches them.

**Files:**
- Create: `services/api/src/lumen_api/sources_db.py` (router)
- Modify: `services/api/src/lumen_api/main.py` (register it)
- Modify: `services/worker/src/lumen_worker/ingest.py` (append `import_tables`)
- Test: `services/api/tests/test_connect_database.py`

**Interfaces:**
- Consumes: `PostgresAdapter` / `MySQLAdapter` (Tasks 12–13), `encrypt_dsn` / `decrypt_dsn` (Task 10), `ensure_tenant_schema` (Task 9).
- Produces: `POST /v1/sources/database`, `GET /v1/sources/{id}/tables`, `POST /v1/sources/{id}/tables` and the `import_tables` job.

- [ ] **Step 1: Write the failing test**

Create `services/api/tests/test_connect_database.py`:

```python
"""Connecting a customer database: structure first, data on demand (D10)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from lumen_api.db.session import user_session
from lumen_api.settings import get_settings
from lumen_api.sources_db import DatabaseSourceCreate, connect_database, list_source_tables
from lumen_api.tenant_db import get_tenant_engine

from tests.test_architect_design import identity  # noqa: F401

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().has_tenant_db, reason="TENANT_DATABASE_URL is not configured"
    ),
]


@pytest.fixture
async def customer_database():
    """A schema on the tenant instance standing in for a customer's own
    database — the adapter cannot tell the difference."""
    name = f"customer_db_{uuid.uuid4().hex[:8]}"
    engine = get_tenant_engine()
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{name}"'))
        await conn.execute(text(f'CREATE TABLE "{name}".people (id text PRIMARY KEY, nm text)'))
        await conn.execute(text(f'CREATE TABLE "{name}".huge (id bigint PRIMARY KEY)'))
        await conn.execute(text(f"INSERT INTO \"{name}\".people VALUES ('p1', 'Ada')"))
    try:
        yield name
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))


async def test_connecting_stores_the_dsn_encrypted(identity, customer_database):
    dsn = get_settings().tenant_database_url.get_secret_value()
    result = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )

    async with user_session(identity.user_id) as db:
        stored = (
            await db.execute(
                text("select dsn_encrypted from public.data_sources where id = :id"),
                {"id": uuid.UUID(result["id"])},
            )
        ).scalar_one()
    assert stored.startswith("v1:")
    assert "postgres" not in stored


async def test_the_dsn_is_never_returned(identity, customer_database):
    dsn = get_settings().tenant_database_url.get_secret_value()
    result = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )
    assert "dsn" not in result
    assert dsn not in str(result)


async def test_structure_is_mirrored_without_copying_data(identity, customer_database):
    """The diagram appears in seconds; no bytes move until a table is
    selected. That is the whole argument for D10."""
    dsn = get_settings().tenant_database_url.get_secret_value()
    created = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )

    tables = await list_source_tables(uuid.UUID(created["id"]), identity)
    names = {t["name"] for t in tables["tables"]}
    assert names == {"people", "huge"}
    assert all(t["imported"] is False for t in tables["tables"])


async def test_the_real_primary_key_is_reported(identity, customer_database):
    dsn = get_settings().tenant_database_url.get_secret_value()
    created = await connect_database(
        DatabaseSourceCreate(name="CRM", kind="postgres", dsn=dsn, schema=customer_database),
        identity,
    )
    tables = await list_source_tables(uuid.UUID(created["id"]), identity)
    people = next(t for t in tables["tables"] if t["name"] == "people")
    assert people["primary_key"] == ["id"]
    assert people["declared"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/api && uv run pytest -q -m integration tests/test_connect_database.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'lumen_api.sources_db'`

- [ ] **Step 3: Write the implementation**

Create `services/api/src/lumen_api/sources_db.py`:

```python
"""Connecting a customer's own database.

Structure first, data on demand (D10). Discovery reads the customer's
information_schema and stores what it found; not a byte is copied until
someone selects a table. That is what makes the diagram appear in seconds
instead of after a 500-table ERP finishes replicating.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text

from lumen.architect.adapters.mysql import MySQLAdapter
from lumen.architect.adapters.postgres import PostgresAdapter
from lumen_api.auth.dependencies import Identity, current_identity, require_role
from lumen_api.credentials import decrypt_dsn, encrypt_dsn
from lumen_api.db.session import user_session
from lumen_api.errors import BadRequest, NotFound
from lumen_api.tenant_db import ensure_tenant_schema

router = APIRouter(prefix="/v1/sources", tags=["sources"])


class DatabaseSourceCreate(BaseModel):
    name: str
    kind: Literal["postgres", "mysql"]
    dsn: str
    schema: str = "public"


class TableSelection(BaseModel):
    tables: list[str]


def _adapter(kind: str, dsn: str, schema: str):
    return PostgresAdapter(dsn, schema) if kind == "postgres" else MySQLAdapter(dsn, schema)


@router.post("/database")
async def connect_database(
    body: DatabaseSourceCreate,
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    """Connect a database and mirror its structure. No data is copied."""
    await ensure_tenant_schema(identity.org_id)

    try:
        structure = await _adapter(body.kind, body.dsn, body.schema).discover()
    except Exception as exc:  # noqa: BLE001 — the customer needs the reason
        raise BadRequest(f"Could not read that database: {exc}") from exc

    discovered = {
        "schema": body.schema,
        "declared": structure.declared,
        "tables": [
            {
                "name": t.name,
                "primary_key": list(t.primary_key or ()),
                "foreign_keys": [list(fk) for fk in t.foreign_keys],
                "columns": [
                    {"name": c.name, "sql_type": c.sql_type.value, "nullable": c.nullable}
                    for c in t.columns
                ],
            }
            for t in structure.tables
        ],
    }

    source_id = uuid.uuid4()
    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "insert into public.data_sources "
                "(id, org_id, name, kind, status, dsn_encrypted, discovered_structure) "
                "values (:id, :org, :name, cast(:kind as public.source_kind), 'idle', "
                "        :dsn, cast(:structure as jsonb))"
            ),
            {
                "id": source_id,
                "org": identity.org_id,
                "name": body.name,
                "kind": body.kind,
                "dsn": encrypt_dsn(body.dsn),
                "structure": json.dumps(discovered),
            },
        )

    # The DSN is never echoed back, here or anywhere else.
    return {"id": str(source_id), "name": body.name, "tables": len(structure.tables)}


@router.get("/{source_id}/tables")
async def list_source_tables(
    source_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    """What this database contains, and which of it has been imported."""
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text(
                    "select discovered_structure, imported_tables "
                    "from public.data_sources where id = :id"
                ),
                {"id": source_id},
            )
        ).mappings().first()
    if row is None or not row["discovered_structure"]:
        raise NotFound("That source has no discovered structure")

    structure = dict(row["discovered_structure"])
    imported = set(row["imported_tables"] or [])
    return {
        "declared": structure.get("declared", False),
        "tables": [
            {**table, "imported": table["name"] in imported}
            for table in structure["tables"]
        ],
    }


@router.post("/{source_id}/tables")
async def select_tables(
    source_id: uuid.UUID,
    body: TableSelection,
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    """Choose which tables to import. Enqueues the copy; returns at once."""
    async with user_session(identity.user_id) as db:
        row = (
            await db.execute(
                text("select discovered_structure from public.data_sources where id = :id"),
                {"id": source_id},
            )
        ).mappings().first()
        if row is None or not row["discovered_structure"]:
            raise NotFound("That source has no discovered structure")

        known = {t["name"] for t in dict(row["discovered_structure"])["tables"]}
        unknown = set(body.tables) - known
        if unknown:
            raise BadRequest(f"Not present on that source: {', '.join(sorted(unknown))}")

        await db.execute(
            text(
                "update public.data_sources set imported_tables = cast(:tables as jsonb) "
                "where id = :id"
            ),
            {"tables": json.dumps(body.tables), "id": source_id},
        )

    from lumen_api.jobs import enqueue_job

    await enqueue_job(
        "import_tables", str(source_id), str(identity.org_id), str(identity.user_id)
    )
    return {"id": str(source_id), "selected": body.tables}
```

This needs two new columns. Create `supabase/migrations/20260807000001_database_sources.sql`:

```sql
-- ADR-0024: a connected customer database is mirrored before it is copied.
-- `discovered_structure` holds what information_schema reported, so the
-- diagram can render a table that exists at origin but has not been
-- imported — without creating an empty placeholder that would make our own
-- information_schema claim a table holding nothing.
alter table public.data_sources
  add column if not exists discovered_structure jsonb,
  add column if not exists imported_tables jsonb not null default '[]'::jsonb;
```

Apply it through the raw asyncpg connection, as every multi-statement migration in this repo is.

Append to `services/worker/src/lumen_worker/ingest.py`:

```python
async def import_tables(
    ctx: dict[str, Any], source_id: str, org_id: str, acting_user_id: str
) -> dict[str, Any]:
    """Copy the selected tables from a connected database into staging."""
    source_uuid, org_uuid, user_uuid = (
        uuid.UUID(source_id), uuid.UUID(org_id), uuid.UUID(acting_user_id)
    )
    await ensure_tenant_schema(org_uuid)

    async with user_session(user_uuid) as db:
        row = (
            await db.execute(
                text(
                    "select kind, dsn_encrypted, discovered_structure, imported_tables "
                    "from public.data_sources where id = :id"
                ),
                {"id": source_uuid},
            )
        ).mappings().first()
    if row is None or not row["dsn_encrypted"]:
        return {"status": "skipped", "reason": "not a connected database source"}

    dsn = decrypt_dsn(row["dsn_encrypted"])
    schema = dict(row["discovered_structure"] or {}).get("schema", "public")
    adapter = (
        PostgresAdapter(dsn, schema) if str(row["kind"]) == "postgres"
        else MySQLAdapter(dsn, schema)
    )
    await adapter.discover()

    connection = get_settings().tenant_database_url.get_secret_value()
    raw = tenant_raw_schema_name(org_uuid)
    copied: dict[str, int] = {}

    for table in list(row["imported_tables"] or []):
        frame = await adapter.read(table)
        materialised = frame.collect() if hasattr(frame, "collect") else frame
        materialised.write_database(
            table_name=f"{raw}.{table}", connection=connection, if_table_exists="replace"
        )
        copied[table] = materialised.height

    await ctx["redis"].enqueue_job("design_schema_job", source_id, org_id, acting_user_id)
    return {"status": "imported", "tables": copied}
```

Extend `ingest.py`'s imports with `decrypt_dsn`, `PostgresAdapter` and `MySQLAdapter`. Register `import_tables` in the worker's function list, and `sources_db.router` in `main.py`'s router list — **including the `app.include_router()` call, not only the import**, which is a mistake this codebase has already made once.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/api && uv run pytest -q -m integration tests/test_connect_database.py`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/api/src/lumen_api/sources_db.py services/api/src/lumen_api/main.py services/worker/src/lumen_worker/ingest.py supabase/migrations/20260807000001_database_sources.sql services/api/tests/test_connect_database.py
git commit -m "feat(architect): connect a customer database, structure first

Discovery reads the customer's information_schema and stores it; not a byte
is copied until a table is selected. That is the whole argument for D10 —
the diagram appears in seconds instead of after a 500-table ERP finishes
replicating, and it shows real declared keys rather than inferred ones.

discovered_structure is stored rather than materialised as empty tables,
because empty placeholders would make our own information_schema claim
tables that hold nothing, which defeats reading structure from the database
in the first place.

The DSN is encrypted on write and never echoed back — a test asserts the
plaintext appears nowhere in the response."
```

---

## Self-review notes

Three requirements were uncovered after Task 23 and are the reason Tasks 24 and 25 exist: `SchemaSpec.layout` was defined but never set, `sanitize_identifier`'s collision suffix produced `users_2` where §3.6 specifies `crm__users`, and nothing reached the Postgres/MySQL adapters that Tasks 12 and 13 built. Spec testing items 11 and 13 were likewise unimplemented.

Two known deviations from the spec, both deliberate:

- **MySQL has no integration test.** No live MySQL exists in this environment, so Task 13 unit-tests the mapping and assembly against recorded `information_schema` rows. A skipped integration test that has never run looks like coverage without being any.
- **Task 3's staging test covers CSV and one rejected format, not all five.** The five formats share one `ReaderFactory` path that its own tests already cover; re-testing each here would test the factory rather than the adapter.

## Done

At this point a customer can upload a file in any supported format or connect their own database, see the data immediately in staging, review a proposed schema with real types, a justified primary key and enforced relationships, accept it, and have a genuine Postgres database built for them — which the engine then analyses in place, and which the Architect maintains as the source changes.

**Not built here, by design:** the screens that show any of this. The raw-data browser and the ER diagram are Project B; the data-health dashboard is Project C. Both were sized on the assumption that this project shipped first, and both get much cheaper because it did — a row browser over a real table is `SELECT … LIMIT/OFFSET`, and the diagram is read from `information_schema` rather than inferred.

