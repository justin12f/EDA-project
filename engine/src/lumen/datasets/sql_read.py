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
