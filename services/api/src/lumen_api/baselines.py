"""Zero-config anomaly baselines — ADR-0013. Every column gets its own
calibrated band, computed from that column's own history, with no
configuration required to produce one (§1). This is an additive step onto
the scheduled tick (ADR-0008) that already reads a source once per run —
nothing here reads the frame a second time.

Baseline kinds all reduce to the *same* rolling `NumericBaseline` (engine
`lumen.sentinel.baseline`) except categorical, which genuinely needs its own
mechanism since a distribution is not one number:

    numeric           the column's own mean, scan to scan (numeric_summary)
    datetime          seconds of lag between "now" and the column's latest
                       value (latest_datetime) — never the raw values
    freeform_string   the column's mean string length (string_length_summary)
    categorical       a rolling frequency distribution (CategoricalBaseline)

Every column's null rate is *also* tracked as its own always-present
`NumericBaseline`, independent of baseline_kind — this is what replaces
ADR-0008's flat 0.05 threshold (§3) with a per-column one, fed into
`lumen.sentinel.detect_drift()`'s optional `null_rate_baselines` parameter.

Scope note: `freeform_string` here is length-distribution only. The ADR's
own "Harder" section already names format-signature inference (character-
class pattern) as "the least precise [kind]... will need the most iteration
against real data" — building a character-class detector before there is
real incident data to iterate it against would be exactly the premature-
investment pattern this file's sibling ADRs keep rejecting (a dedicated
analytics store in ADR-0004, streaming CDC in ADR-0008, Option D's learned
models in this same ADR). Length alone still catches the ADR's own coarse
example (an email column suddenly full of much shorter or longer values);
a corruption that happens to preserve length is the acknowledged gap.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from lumen.datasets.materialize import (
    column_values,
    latest_datetime,
    numeric_summary,
    string_length_summary,
    value_counts,
)
from lumen.sentinel import (
    DEFAULT_CATEGORY_DEVIATION,
    DEFAULT_NEW_CATEGORY_SHARE,
    DEFAULT_Z,
    MIN_SAMPLE_SIZE,
    CategoricalBaseline,
    NumericBaseline,
    categorical_shift,
    classify_baseline_kind,
)
from lumen_api.auth.dependencies import Identity, current_identity, require_role
from lumen_api.db.session import user_session
from lumen_api.errors import NotFound


def _numeric_to_json(baseline: NumericBaseline) -> dict[str, Any]:
    return {"mean": baseline.mean, "variance": baseline.variance, "sample_size": baseline.sample_size}


def _numeric_from_json(data: dict[str, Any] | None) -> NumericBaseline:
    if not data:
        return NumericBaseline()
    return NumericBaseline(
        mean=data.get("mean", 0.0), variance=data.get("variance", 0.0), sample_size=data.get("sample_size", 0)
    )


def _categorical_to_json(baseline: CategoricalBaseline) -> dict[str, Any]:
    return {"frequencies": baseline.frequencies, "sample_size": baseline.sample_size}


def _categorical_from_json(data: dict[str, Any] | None) -> CategoricalBaseline:
    if not data:
        return CategoricalBaseline()
    return CategoricalBaseline(frequencies=dict(data.get("frequencies", {})), sample_size=data.get("sample_size", 0))


def _lag_seconds(latest_value: str, now: datetime) -> float | None:
    try:
        parsed = datetime.fromisoformat(latest_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (now - parsed).total_seconds())


async def _existing_column_baselines(db: AsyncSession, source_id: uuid.UUID) -> dict[str, dict[str, Any]]:
    rows = (
        await db.execute(
            text(
                "select column_name, baseline_kind, model, override "
                "from public.column_baselines where source_id = :source"
            ),
            {"source": source_id},
        )
    ).mappings().all()
    return {row["column_name"]: dict(row) for row in rows}


async def prepare_and_update_column_baselines(
    db: AsyncSession,
    org_id: uuid.UUID,
    source_id: uuid.UUID,
    frame: Any,
    backend: str,
    schema: dict[str, str],
    row_count: int,
    null_rates: dict[str, float],
) -> tuple[dict[str, NumericBaseline], list[dict[str, Any]]]:
    """Classifies and updates every column's baseline for this scan, and
    returns `(null_rate_baselines, findings)`:

    - `null_rate_baselines` is the state *before* this scan folded in —
      exactly what `detect_drift(..., null_rate_baselines=...)` needs, so a
      comparison is never made against a band this same scan just widened
      to include itself.
    - `findings` is this scan's *value*-dimension breaches only (numeric,
      categorical, datetime, freeform_string) — null-rate breaches are
      `detect_drift()`'s own job via the baselines returned above; this
      function does not duplicate that check.

    Every row this touches is updated and persisted before returning; the
    caller writes nothing itself.
    """
    existing = await _existing_column_baselines(db, source_id)
    null_rate_baselines = {
        column: _numeric_from_json(row["model"].get("null_rate")) for column, row in existing.items()
    }

    columns = list(schema)
    kinds: dict[str, str] = {}
    for column in columns:
        # Classified once, on this column's first-ever scan, and reused
        # from then on — a low-row-count scan's own cardinality ratio is
        # noisy (a 6-row sample of a genuinely categorical column can look
        # high-cardinality by chance), and reclassifying every scan would
        # silently discard that column's whole calibration history the
        # moment it did, since a numeric-shaped model and a
        # categorical-shaped one share no fields `_numeric_from_json`/
        # `_categorical_from_json` can recover from each other.
        stored_kind = existing.get(column, {}).get("baseline_kind")
        if stored_kind:
            kinds[column] = stored_kind
            continue
        non_null = round(row_count * (1 - null_rates.get(column, 0.0)))
        distinct = len(column_values(frame, backend, column)) if non_null else 0
        kinds[column] = classify_baseline_kind(schema[column], non_null_count=non_null, distinct_count=distinct)

    numeric_columns = [c for c in columns if kinds[c] == "numeric"]
    categorical_columns = [c for c in columns if kinds[c] == "categorical"]
    freeform_columns = [c for c in columns if kinds[c] == "freeform_string"]
    datetime_columns = [c for c in columns if kinds[c] == "datetime"]

    numeric_stats = numeric_summary(frame, backend, numeric_columns) if numeric_columns else {}
    categorical_counts = value_counts(frame, backend, categorical_columns) if categorical_columns else {}
    freeform_stats = string_length_summary(frame, backend, freeform_columns) if freeform_columns else {}
    datetime_latest = latest_datetime(frame, backend, datetime_columns) if datetime_columns else {}

    now = datetime.now(timezone.utc)
    findings: list[dict[str, Any]] = []

    for column in columns:
        kind = kinds[column]
        row = existing.get(column, {})
        override: dict[str, Any] = row.get("override") or {}
        prior_model: dict[str, Any] = row.get("model") or {}

        if kind == "categorical":
            prior_value = _categorical_from_json(prior_model.get("value"))
            counts = categorical_counts.get(column, {})
            total = sum(counts.values())
            shifts = categorical_shift(
                prior_value,
                counts,
                total,
                deviation_threshold=override.get("deviation_threshold", DEFAULT_CATEGORY_DEVIATION),
                new_category_share=override.get("new_category_share", DEFAULT_NEW_CATEGORY_SHARE),
            )
            for shift in shifts:
                findings.append(
                    {
                        "column": column,
                        "baseline_kind": kind,
                        "check": "value",
                        "shift_kind": shift.kind,
                        "category": shift.category,
                        "current_share": shift.current_share,
                        "baseline_share": shift.baseline_share,
                    }
                )
            updated_value: NumericBaseline | CategoricalBaseline = prior_value.updated(counts, total)
            value_json = _categorical_to_json(updated_value)
        else:
            prior_value = _numeric_from_json(prior_model.get("value"))
            observation: float | None = None
            if kind == "numeric":
                stat = numeric_stats.get(column)
                observation = stat["mean"] if stat and stat["count"] else None
            elif kind == "freeform_string":
                stat = freeform_stats.get(column)
                observation = stat["mean"] if stat and stat["count"] else None
            elif kind == "datetime":
                latest = datetime_latest.get(column)
                observation = _lag_seconds(latest, now) if latest else None

            if observation is not None:
                z = override.get("z", DEFAULT_Z)
                if not prior_value.is_within_band(observation, z=z):
                    findings.append(
                        {
                            "column": column,
                            "baseline_kind": kind,
                            "check": "value",
                            "observed": observation,
                            "baseline_mean": prior_value.mean,
                            "baseline_stdev": prior_value.stdev,
                        }
                    )
                updated_value = prior_value.updated(observation)
            else:
                updated_value = prior_value
            value_json = _numeric_to_json(updated_value)

        prior_null = null_rate_baselines.get(column, NumericBaseline())
        updated_null = prior_null.updated(null_rates.get(column, 0.0))

        await db.execute(
            text(
                "insert into public.column_baselines "
                "(org_id, source_id, column_name, baseline_kind, model, sample_size, computed_at) "
                "values (:org, :source, :column, :kind, cast(:model as jsonb), :sample, now()) "
                "on conflict (source_id, column_name) do update set "
                "  baseline_kind = :kind, model = cast(:model as jsonb), "
                "  sample_size = :sample, computed_at = now()"
            ),
            {
                "org": org_id,
                "source": source_id,
                "column": column,
                "kind": kind,
                "model": json.dumps({"value": value_json, "null_rate": _numeric_to_json(updated_null)}),
                "sample": updated_null.sample_size,
            },
        )

    return null_rate_baselines, findings


async def compute_and_remember_source_baseline(
    db: AsyncSession, org_id: uuid.UUID, source_id: uuid.UUID, row_count: int
) -> dict[str, Any] | None:
    """Row-count *delta* (this scan's row count minus the last scan's), not
    the raw level — a healthy, steadily growing source has a level that
    keeps climbing every scan, which breaks the "normal is roughly stable"
    assumption a control-limit band relies on. The delta is naturally closer
    to stationary: a source growing ~120 rows a day has a delta baseline
    centered near 120, and a stalled upstream job (delta drops toward 0) is
    exactly "the table just stopped growing" — ADR-0013 §2's own motivating
    example, invisible to every per-column check since nothing about an
    unchanged column's own values looks wrong.
    """
    row = (
        await db.execute(
            text("select model from public.source_baselines where source_id = :source"), {"source": source_id}
        )
    ).mappings().first()
    model: dict[str, Any] = (row["model"] if row else None) or {}
    last_row_count = model.get("last_row_count")
    baseline = _numeric_from_json(model.get("delta"))

    finding: dict[str, Any] | None = None
    if last_row_count is not None:
        delta = row_count - last_row_count
        if not baseline.is_within_band(delta):
            finding = {
                "check": "volume",
                "observed_delta": delta,
                "baseline_mean": baseline.mean,
                "baseline_stdev": baseline.stdev,
                "row_count": row_count,
                "previous_row_count": last_row_count,
            }
        baseline = baseline.updated(delta)

    await db.execute(
        text(
            "insert into public.source_baselines (org_id, source_id, model, sample_size, computed_at) "
            "values (:org, :source, cast(:model as jsonb), :sample, now()) "
            "on conflict (source_id) do update set "
            "  model = cast(:model as jsonb), sample_size = :sample, computed_at = now()"
        ),
        {
            "org": org_id,
            "source": source_id,
            "model": json.dumps({"delta": _numeric_to_json(baseline), "last_row_count": row_count}),
            "sample": baseline.sample_size,
        },
    )
    return finding


# ── reading baselines and setting a manual override (ADR-0013 §4, Action Item 6) ──
#
# Read access matches every other Sentinel-adjacent GET in this codebase
# (current_identity — a viewer can watch, same as the drift feed). Write
# access matches schedules.py's own PUT: "any member" — a per-column
# threshold override is the same class of operational tuning as the
# schedule toggle it lives right next to, not an owner-only setting.

router = APIRouter(prefix="/v1/sources/{source_id}/baselines", tags=["baselines"])


class OverrideUpdate(BaseModel):
    z: float | None = None
    deviation_threshold: float | None = None
    new_category_share: float | None = None


@router.get("")
async def list_baselines(
    source_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        columns = (
            await db.execute(
                text(
                    "select column_name, baseline_kind, sample_size, override, computed_at "
                    "from public.column_baselines where source_id = :source order by column_name"
                ),
                {"source": source_id},
            )
        ).mappings().all()
        source_row = (
            await db.execute(
                text("select sample_size, computed_at from public.source_baselines where source_id = :source"),
                {"source": source_id},
            )
        ).mappings().first()

    return {
        "columns": [
            {
                "column": row["column_name"],
                "baseline_kind": row["baseline_kind"],
                "sample_size": row["sample_size"],
                "calibrated": row["sample_size"] >= MIN_SAMPLE_SIZE,
                "override": row["override"],
                "computed_at": row["computed_at"],
            }
            for row in columns
        ],
        "source": (
            {
                "sample_size": source_row["sample_size"],
                "calibrated": source_row["sample_size"] >= MIN_SAMPLE_SIZE,
                "computed_at": source_row["computed_at"],
            }
            if source_row
            else None
        ),
        "min_sample_size": MIN_SAMPLE_SIZE,
    }


@router.put("/{column_name}/override")
async def set_override(
    source_id: uuid.UUID,
    column_name: str,
    body: OverrideUpdate,
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    override = {key: value for key, value in body.model_dump().items() if value is not None}
    async with user_session(identity.user_id) as db:
        result = await db.execute(
            text(
                "update public.column_baselines set override = cast(:override as jsonb) "
                "where source_id = :source and column_name = :column"
            ),
            {"override": json.dumps(override) if override else None, "source": source_id, "column": column_name},
        )
    if result.rowcount == 0:
        raise NotFound(f"No baseline yet for column '{column_name}' on this source")
    return {"column": column_name, "override": override or None}


@router.delete("/{column_name}/override")
async def clear_override(
    source_id: uuid.UUID,
    column_name: str,
    identity: Annotated[Identity, Depends(require_role("member"))],
) -> dict[str, Any]:
    async with user_session(identity.user_id) as db:
        result = await db.execute(
            text(
                "update public.column_baselines set override = null "
                "where source_id = :source and column_name = :column"
            ),
            {"source": source_id, "column": column_name},
        )
    if result.rowcount == 0:
        raise NotFound(f"No baseline yet for column '{column_name}' on this source")
    return {"column": column_name, "override": None}
