"""ArtifactDependency: a lineage registry, not a graph engine (ADR-0010).

"Who reads what," written transactionally by the code that already knows the
dependency at creation time — never inferred later by parsing a spec. Two
write sites today: a source's currently-applied pipeline
(`apply_pipeline.py`) and an approved canonical entity's members
(`proposals.py`'s entity_mapping accept). `analysis_widget` is a reserved
`artifact_kind` with no writer yet — see the migration's own note on why.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def touched_columns(steps: list[dict[str, Any]], schema: dict[str, str]) -> list[str]:
    """Which columns a pipeline reads or writes, for the lineage registry.

    A step that names `columns` in its kwargs (the standard scoping
    convention — see `ColumnScopedStep`) is scoped to exactly those; a step
    that doesn't (a row-wide operation like `remove_duplicates_rows`) is
    treated as depending on the *whole* schema. That is a deliberate
    over-approximation: a row-wide step's output can be affected by any
    column changing, and understating that is the failure mode that would
    make an impact report silently miss something (ADR-0010 §4's whole
    point — a truncated report has to say so, not just be wrong quietly).
    """
    touched: set[str] = set()
    for step in steps:
        if not step:
            continue
        kwargs = next(iter(step.values()), None) or {}
        columns = kwargs.get("columns")
        if columns:
            touched.update(columns)
        else:
            touched.update(schema)
    return sorted(touched)


async def replace_pipeline_dependency(
    db: AsyncSession,
    org_id: uuid.UUID,
    source_id: uuid.UUID,
    artifact_id: uuid.UUID,
    columns: list[str],
) -> None:
    """A source has exactly one *current* pipeline — replace, don't
    accumulate, so this table reflects what is actually running rather than
    its full history. Same session as the apply itself, so a failed apply
    never leaves a dependency row for a pipeline that never took effect.
    """
    await db.execute(
        text(
            "delete from public.artifact_dependencies "
            "where org_id = :org and artifact_kind = 'pipeline' and source_id = :source"
        ),
        {"org": org_id, "source": source_id},
    )
    if not columns:
        return
    await db.execute(
        text(
            "insert into public.artifact_dependencies "
            "(org_id, artifact_kind, artifact_id, source_id, columns) "
            "values (:org, 'pipeline', :artifact, :source, :columns)"
        ),
        {"org": org_id, "artifact": artifact_id, "source": source_id, "columns": columns},
    )


async def record_entity_dependency(
    db: AsyncSession,
    org_id: uuid.UUID,
    entity_id: uuid.UUID,
    source_id: uuid.UUID,
    column: str,
) -> None:
    """One row per member — a `CanonicalEntity` is never revised in place
    (a changed mapping is a new proposal), so unlike the pipeline case there
    is nothing to replace."""
    await db.execute(
        text(
            "insert into public.artifact_dependencies "
            "(org_id, artifact_kind, artifact_id, source_id, columns) "
            "values (:org, 'canonical_entity', :artifact, :source, :columns)"
        ),
        {"org": org_id, "artifact": entity_id, "source": source_id, "columns": [column]},
    )
