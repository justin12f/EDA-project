"""Conversion funnel analysis with drop-off rates and bottleneck detection."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `BusinessStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FunnelStage:
    """Immutable statistics for a single funnel stage."""

    stage_name: str
    stage_index: int
    n_users: int
    conversion_rate_from_top: float
    step_conversion_rate: float
    drop_off_rate: float
    drop_off_count: int
    is_bottleneck: bool

class FunnelMetricsComputer:
    """Computes conversion and drop-off rates for each funnel stage."""

    def compute(
        self,
        stage_counts: list[tuple[str, int]],
        bottleneck_threshold: float,
    ) -> list[FunnelStage]:
        """Compute funnel metrics for all stages.

        Args:
            stage_counts: Ordered list of (stage_name, user_count) tuples.
            bottleneck_threshold: Drop-off rate above which a stage is a bottleneck.

        Returns:
            List of FunnelStage objects.
        """
        top_count = stage_counts[0][1]
        stages: list[FunnelStage] = []

        for i, (name, count) in enumerate(stage_counts):
            conv_from_top = count / top_count if top_count > 0 else 0.0
            prev_count = stage_counts[i - 1][1] if i > 0 else count
            step_conv = count / prev_count if prev_count > 0 else 1.0
            drop_off = 1.0 - step_conv
            drop_off_count = prev_count - count if i > 0 else 0

            stages.append(
                FunnelStage(
                    stage_name=name,
                    stage_index=i,
                    n_users=count,
                    conversion_rate_from_top=round(conv_from_top, 6),
                    step_conversion_rate=round(step_conv, 6),
                    drop_off_rate=round(drop_off, 6),
                    drop_off_count=max(drop_off_count, 0),
                    is_bottleneck=(i > 0 and drop_off > bottleneck_threshold),
                )
            )

        return stages

class FunnelFromEventsBuilder:
    """Builds funnel counts from an event log DataFrame.

    Counts unique users who reached each stage, enforcing that a user
    must pass through all prior stages (sequential funnel logic).
    """

    def build(
        self,
        data_frame: pd.DataFrame,
        user_column: str,
        event_column: str,
        stage_order: list[str],
    ) -> list[tuple[str, int]]:
        """Build ordered stage counts from event log.

        Args:
            data_frame: Event log (one row per user-event).
            user_column: User identifier column.
            event_column: Event type column.
            stage_order: Ordered list of event names defining the funnel.

        Returns:
            Ordered list of (stage_name, unique_user_count) tuples.

        Raises:
            ValueError: If no users match the first stage.
        """
        stage_users: list[set] = []

        for i, stage in enumerate(stage_order):
            stage_set = set(
                data_frame[data_frame[event_column] == stage][user_column].unique()
            )
            if i > 0 and stage_users:
                # Sequential: only users who passed the previous stage
                stage_set = stage_set & stage_users[i - 1]
            stage_users.append(stage_set)

        if not stage_users or len(stage_users[0]) == 0:
            raise ValueError(
                f"No users found at funnel entry stage '{stage_order[0]}'. "
                "Check stage_order and event_column values."
            )

        return [(stage, len(users)) for stage, users in zip(stage_order, stage_users)]

class ConversionFunnelCalculator:
    """Conversion funnel analysis from aggregated counts or event logs.

    Workflow — from pre-aggregated counts:
        calculator = ConversionFunnelCalculator()
        result = calculator.calculate(
            stage_counts={
                "Visit": 10000,
                "Sign Up": 3000,
                "Activate": 1500,
                "Purchase": 500,
            },
            bottleneck_threshold=0.5,
        )

    Workflow — from event log DataFrame:
        result = calculator.calculate(
            data_frame=df,
            user_column="user_id",
            event_column="event_name",
            stage_order=["Visit", "Sign Up", "Activate", "Purchase"],
            bottleneck_threshold=0.5,
        )
    """

    _MINIMUM_STAGES: int = 2

    def __init__(self) -> None:
        self._metrics_computer = FunnelMetricsComputer()
        self._events_builder = FunnelFromEventsBuilder()

    def calculate(
        self,
        stage_counts: dict[str, int] | None = None,
        data_frame: pd.DataFrame | None = None,
        user_column: str | None = None,
        event_column: str | None = None,
        stage_order: list[str] | None = None,
        bottleneck_threshold: float = 0.5,
    ) -> dict:
        """Compute funnel metrics.

        Args:
            stage_counts: Pre-aggregated {stage_name: count} dict (ordered).
            data_frame: Event log DataFrame (alternative to stage_counts).
            user_column: User identifier column (event log mode).
            event_column: Event type column (event log mode).
            stage_order: Ordered funnel stages (event log mode).
            bottleneck_threshold: Drop-off rate above which a stage is flagged.

        Returns:
            Dict with per-stage metrics, bottlenecks, and overall conversion.

        Raises:
            ValueError: If neither stage_counts nor event log params are provided,
                or if fewer than 2 stages are defined.
        """
        if not 0.0 < bottleneck_threshold < 1.0:
            raise ValueError(
                f"bottleneck_threshold must be in (0, 1). Got {bottleneck_threshold}."
            )

        if stage_counts is not None:
            ordered_counts = list(stage_counts.items())
        elif data_frame is not None and user_column and event_column and stage_order:
            for col in (user_column, event_column):
                if col not in data_frame.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame.")
            ordered_counts = self._events_builder.build(
                data_frame, user_column, event_column, stage_order
            )
        else:
            raise ValueError(
                "Provide either 'stage_counts' or all of: "
                "'data_frame', 'user_column', 'event_column', 'stage_order'."
            )

        if len(ordered_counts) < self._MINIMUM_STAGES:
            raise ValueError(
                f"At least {self._MINIMUM_STAGES} funnel stages required. "
                f"Got {len(ordered_counts)}."
            )

        stages = self._metrics_computer.compute(ordered_counts, bottleneck_threshold)
        bottlenecks = [s.stage_name for s in stages if s.is_bottleneck]
        overall_conversion = stages[-1].conversion_rate_from_top

        return {
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "stage_index": s.stage_index,
                    "n_users": s.n_users,
                    "conversion_rate_from_top": s.conversion_rate_from_top,
                    "step_conversion_rate": s.step_conversion_rate,
                    "drop_off_rate": s.drop_off_rate,
                    "drop_off_count": s.drop_off_count,
                    "is_bottleneck": s.is_bottleneck,
                }
                for s in stages
            ],
            "overall_conversion_rate": round(overall_conversion, 6),
            "overall_conversion_pct": round(overall_conversion * 100, 4),
            "bottleneck_stages": bottlenecks,
            "n_stages": len(stages),
            "top_of_funnel_users": ordered_counts[0][1],
            "bottom_of_funnel_users": ordered_counts[-1][1],
            "bottleneck_threshold": bottleneck_threshold,
        }
