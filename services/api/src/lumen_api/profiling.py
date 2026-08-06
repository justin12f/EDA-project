"""What "profile a dataset" means — one implementation, two callers: the chat
tool (`agents/registry.py`'s `profile_source`) and the scheduled tick
(ADR-0008's worker). Duplicating this would let the two profiles drift apart,
which defeats the Sentinel's whole premise — it diffs a later scan against
exactly what this function wrote, so both call sites have to write the same
shape or the diff is comparing things that were never really the same measure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from lumen.datasets.materialize import duplicate_counts, null_rates
from lumen_api.context.store import ContextEntry, ContextStore, Kind
from lumen_api.datasets.store import DatasetHandle

# Columns worth checking for duplicates: an id-like or hash-like name. Checking
# every column on a wide frame costs more than it tells you.
DUPLICATE_HINTS = ("id", "hash", "email", "key", "code", "uuid")


@dataclass(frozen=True)
class ProfileResult:
    null_rates: dict[str, float]
    duplicates: dict[str, int]
    summary: str


async def profile_and_remember(
    memory: ContextStore, handle: DatasetHandle, frame: Any, rid: str
) -> ProfileResult:
    """Compute null rates and duplicate counts, and write the org-scoped fact
    both callers rely on: the chat tool so a teammate doesn't re-audit an
    already-audited source, the Sentinel so a later scan has something to
    diff against (`ContextStore.latest_profile`)."""
    rates = null_rates(frame, handle.backend)
    candidates = [
        column for column in handle.schema if any(hint in column.lower() for hint in DUPLICATE_HINTS)
    ]
    duplicates = duplicate_counts(frame, handle.backend, candidates) if candidates else {}

    notable = {c: r for c, r in rates.items() if r > 0.005}
    dupes = {k: v for k, v in duplicates.items() if v > 0}

    summary = (
        f"{handle.row_count} rows, {len(handle.schema)} columns. "
        + (
            "Null rates above 0.5%: "
            + ", ".join(f"{c} {r * 100:.1f}%" for c, r in sorted(notable.items()))
            if notable
            else "No column exceeds a 0.5% null rate."
        )
        + (
            " Duplicates: " + ", ".join(f"{c} x{v}" for c, v in sorted(dupes.items()))
            if dupes
            else ""
        )
    )

    await memory.remember(
        ContextEntry(
            kind=Kind.PROFILE,
            title=f"Profile of {handle.label or rid}",
            content=summary,
            source_id=handle.source_id,
            rid=rid,
            # `schema` alongside the numbers this entry was already written
            # for: it is the baseline a later scan diffs against (ADR-0008),
            # not a second profiling pass.
            metadata={"schema": handle.schema, "null_rates": rates, "duplicates": dupes},
        )
    )

    return ProfileResult(null_rates=rates, duplicates=dupes, summary=summary)


async def remember_columns(
    memory: ContextStore, handle: DatasetHandle, columns: list[str]
) -> list[tuple[uuid.UUID, str, str]]:
    """Write one org-scoped, embedded fact per column in `columns` — not the
    whole schema, just what's new or changed since the last tick. Returns
    `(entry_id, column, embedded_content)` triples so a caller doing a
    follow-up similarity search (ADR-0009's clustering) reuses the exact
    string that was embedded, instead of reconstructing it and risking the
    two drifting apart.

    `Kind.SCHEMA` was reserved in the `context_kind` enum from ADR-0006 but
    never used until now: it is exactly "a fact about one column," which is
    what ADR-0009's entity-resolution clustering searches across sources for.
    Scoping to changed columns only, rather than re-embedding a source's
    whole schema every tick, is what keeps "piggyback on the existing worker
    tick at no new cost" (the ADR's own framing) actually true.
    """
    contents = [
        f"{humanize_column(column)} ({handle.schema.get(column, 'unknown')})" for column in columns
    ]
    entries = [
        ContextEntry(
            kind=Kind.SCHEMA,
            title=f"{handle.label or handle.rid}.{column}",
            content=content,
            source_id=handle.source_id,
            rid=handle.rid,
            metadata={"column": column, "dtype": handle.schema.get(column)},
        )
        for column, content in zip(columns, contents, strict=True)
    ]
    ids = await memory.remember_many(entries)
    return list(zip(ids, columns, contents, strict=True))


def humanize_column(column: str) -> str:
    return column.replace("_", " ").replace("-", " ").strip() or column
