"""Replay a candidate cleaning pipeline against cached data, write nothing.

ADR-0008's confidence signal: before a `pipeline_patch` proposal is shown to
anyone, this runs it against the source's current dataset handle and reports
what changed, without materialising a result. ADR-0010 already plans to
extract this into shared infrastructure for simulating a proposal's impact on
*other* artifacts, not just validating the proposal itself — built as its own
module from the start for exactly that reason, not bundled into the Sentinel
agent that happens to be its first caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumen.data_cleaning.data_cleaning_pipeline import PipelineBuilder
from lumen.datasets.materialize import frame_schema, null_rates, row_count
from lumen_api.datasets.store import HandleStore


@dataclass(frozen=True)
class ShadowRunResult:
    ok: bool
    error: str | None = None
    row_count_before: int = 0
    row_count_after: int = 0
    null_rates_before: dict[str, float] | None = None
    null_rates_after: dict[str, float] | None = None
    schema_after: dict[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "row_count_before": self.row_count_before,
            "row_count_after": self.row_count_after,
            "null_rates_before": self.null_rates_before,
            "null_rates_after": self.null_rates_after,
            "schema_after": self.schema_after,
        }


async def shadow_run(store: HandleStore, rid: str, steps: list[dict[str, Any]]) -> ShadowRunResult:
    """Build and run a candidate pipeline against `rid`'s current data, in
    memory. Never calls `store.put()` — a shadow run that materialised a
    handle would not be a shadow run, it would be applying the patch to find
    out whether the patch was safe to apply.
    """
    try:
        handle = await store.get(rid)
        frame = await store.resolve(rid)
    except Exception as exc:  # noqa: BLE001 — a bad rid is a shadow-run failure, not a crash
        return ShadowRunResult(ok=False, error=f"could not load '{rid}': {exc}")

    before_rates = null_rates(frame, handle.backend)
    before_count = row_count(frame, handle.backend)

    try:
        pipeline = PipelineBuilder(frame).build(steps)
        cleaned = pipeline.run(frame)
    except Exception as exc:  # noqa: BLE001 — this *is* the low-confidence signal, not a crash
        return ShadowRunResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            row_count_before=before_count,
            null_rates_before=before_rates,
        )

    return ShadowRunResult(
        ok=True,
        row_count_before=before_count,
        row_count_after=row_count(cleaned, handle.backend),
        null_rates_before=before_rates,
        null_rates_after=null_rates(cleaned, handle.backend),
        schema_after=frame_schema(cleaned, handle.backend),
    )
