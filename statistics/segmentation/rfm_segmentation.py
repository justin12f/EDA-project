"""RFM (Recency, Frequency, Monetary) segmentation with scoring and labeling."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `SegmentationStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RFMRecord:
    """Immutable RFM record for a single customer/entity."""

    entity_id: str
    recency: float
    frequency: int
    monetary: float
    r_score: int
    f_score: int
    m_score: int
    rfm_score: int
    segment: str

class RFMMetricsComputer:
    """Computes raw RFM metrics from a transactional DataFrame.

    Recency:   Days since most recent transaction (lower = better).
    Frequency: Number of distinct transactions.
    Monetary:  Total transaction value.
    """

    def compute(
        self,
        data_frame: pd.DataFrame,
        customer_column: str,
        date_column: str,
        amount_column: str,
        reference_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Aggregate RFM metrics per customer.

        Args:
            data_frame: Transactional DataFrame.
            customer_column: Customer/entity identifier column.
            date_column: Transaction date column.
            amount_column: Transaction amount column.
            reference_date: Snapshot date for recency computation.

        Returns:
            DataFrame with columns: entity_id, recency, frequency, monetary.
        """
        df = data_frame.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        rfm = (
            df.groupby(customer_column)
            .agg(
                recency=(date_column, lambda x: (reference_date - x.max()).days),
                frequency=(date_column, "count"),
                monetary=(amount_column, "sum"),
            )
            .reset_index()
        )
        rfm.columns = ["entity_id", "recency", "frequency", "monetary"]
        return rfm

class QuantileRFMScorer:
    """Assigns 1-5 scores for each RFM dimension using quintile bucketing.

    Recency scoring is inverted: lower recency (more recent) = higher score.
    Frequency and Monetary: higher = higher score.

    Ties are broken using 'first' rank to ensure all 5 quintiles
    are populated even with duplicate values.
    """

    _N_QUINTILES: int = 5

    def score(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Assign R, F, M scores via quintile assignment.

        Args:
            rfm: DataFrame with recency, frequency, monetary columns.

        Returns:
            DataFrame with r_score, f_score, m_score columns added.
        """
        scored = rfm.copy()

        # Recency: lower recency = higher score (reverse ranking)
        scored["r_score"] = pd.qcut(
            scored["recency"].rank(method="first", ascending=False),
            q=self._N_QUINTILES,
            labels=list(range(1, self._N_QUINTILES + 1)),
        ).astype(int)

        # Frequency: higher = better
        scored["f_score"] = pd.qcut(
            scored["frequency"].rank(method="first"),
            q=self._N_QUINTILES,
            labels=list(range(1, self._N_QUINTILES + 1)),
        ).astype(int)

        # Monetary: higher = better
        scored["m_score"] = pd.qcut(
            scored["monetary"].rank(method="first"),
            q=self._N_QUINTILES,
            labels=list(range(1, self._N_QUINTILES + 1)),
        ).astype(int)

        scored["rfm_score"] = (
            scored["r_score"] * 100
            + scored["f_score"] * 10
            + scored["m_score"]
        )

        return scored

class RFMSegmentAssigner:
    """Assigns business segment labels based on R and F scores.

    Segment matrix (standard RFM taxonomy):

        R=5, F=5 → Champions
        R=4-5, F=4-5 → Loyal Customers
        R=4-5, F=1-3 → Potential Loyalists
        R=5, F=1 → New Customers
        R=3-4, F=3-5 → Promising
        R=3, F=3 → Need Attention
        R=2-3, F=1-2 → About to Sleep
        R=1-2, F=3-5 → At Risk
        R=1-2, F=1-2 → Lost
    """

    _SEGMENT_RULES: list[tuple[tuple[int, int], tuple[int, int], str]] = [
        ((5, 5), (5, 5),  "Champions"),
        ((4, 5), (4, 5),  "Loyal Customers"),
        ((4, 5), (1, 3),  "Potential Loyalists"),
        ((5, 5), (1, 1),  "New Customers"),
        ((3, 4), (3, 5),  "Promising"),
        ((3, 3), (3, 3),  "Need Attention"),
        ((2, 3), (1, 2),  "About to Sleep"),
        ((1, 2), (3, 5),  "At Risk"),
        ((1, 2), (1, 2),  "Lost"),
    ]

    def assign(self, r_score: int, f_score: int) -> str:
        """Assign segment label for given R and F scores.

        Args:
            r_score: Recency score (1-5).
            f_score: Frequency score (1-5).

        Returns:
            Segment label string.
        """
        for (r_min, r_max), (f_min, f_max), segment in self._SEGMENT_RULES:
            if r_min <= r_score <= r_max and f_min <= f_score <= f_max:
                return segment
        return "Uncategorized"

class RFMSegmentationCalculator:
    """Full RFM segmentation pipeline from transactional data.

    Workflow:
        calculator = RFMSegmentationCalculator()
        result = calculator.calculate(
            data_frame=df,
            customer_column="customer_id",
            date_column="purchase_date",
            amount_column="order_value",
            reference_date=None,    # optional, defaults to max date
        )
    """

    _MINIMUM_TRANSACTIONS: int = 10

    def __init__(self) -> None:
        self._metrics_computer = RFMMetricsComputer()
        self._scorer = QuantileRFMScorer()
        self._assigner = RFMSegmentAssigner()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        customer_column: str,
        date_column: str,
        amount_column: str,
        reference_date: pd.Timestamp | None = None,
    ) -> dict:
        """Run full RFM segmentation.

        Args:
            data_frame: Transactional DataFrame (one row per transaction).
            customer_column: Customer/entity identifier column.
            date_column: Transaction date column (parseable by pd.to_datetime).
            amount_column: Transaction monetary amount column.
            reference_date: Snapshot date for recency. Defaults to max date + 1 day.

        Returns:
            Dict with per-customer RFM scores, segments, and segment summary.

        Raises:
            KeyError: If required columns are not found.
            ValueError: If data is insufficient.
        """
        required = [customer_column, date_column, amount_column]
        missing = [c for c in required if c not in data_frame.columns]
        if missing:
            raise KeyError(f"Required columns not found: {missing}")

        if len(data_frame) < self._MINIMUM_TRANSACTIONS:
            raise ValueError(
                f"At least {self._MINIMUM_TRANSACTIONS} transactions required. "
                f"Got {len(data_frame)}."
            )

        clean = data_frame[required].dropna()
        ref_date = (
            reference_date
            if reference_date is not None
            else pd.to_datetime(clean[date_column]).max() + pd.Timedelta(days=1)
        )

        rfm = self._metrics_computer.compute(
            clean, customer_column, date_column, amount_column, ref_date
        )
        scored = self._scorer.score(rfm)

        records: list[RFMRecord] = [
            RFMRecord(
                entity_id=str(row["entity_id"]),
                recency=float(row["recency"]),
                frequency=int(row["frequency"]),
                monetary=float(row["monetary"]),
                r_score=int(row["r_score"]),
                f_score=int(row["f_score"]),
                m_score=int(row["m_score"]),
                rfm_score=int(row["rfm_score"]),
                segment=self._assigner.assign(int(row["r_score"]), int(row["f_score"])),
            )
            for _, row in scored.iterrows()
        ]

        segment_counts: dict[str, int] = {}
        segment_monetary: dict[str, float] = {}
        for r in records:
            segment_counts[r.segment] = segment_counts.get(r.segment, 0) + 1
            segment_monetary[r.segment] = (
                segment_monetary.get(r.segment, 0.0) + r.monetary
            )

        total_monetary = sum(segment_monetary.values())

        return {
            "customers": [
                {
                    "entity_id": r.entity_id,
                    "recency": r.recency,
                    "frequency": r.frequency,
                    "monetary": round(r.monetary, 2),
                    "r_score": r.r_score,
                    "f_score": r.f_score,
                    "m_score": r.m_score,
                    "rfm_score": r.rfm_score,
                    "segment": r.segment,
                }
                for r in records
            ],
            "segment_summary": {
                segment: {
                    "count": count,
                    "proportion": round(count / len(records), 4),
                    "total_monetary": round(segment_monetary[segment], 2),
                    "monetary_share": round(
                        segment_monetary[segment] / total_monetary, 4
                    ) if total_monetary > 0 else 0.0,
                }
                for segment, count in sorted(
                    segment_counts.items(), key=lambda x: x[1], reverse=True
                )
            },
            "n_customers": len(records),
            "n_transactions": len(clean),
            "reference_date": str(ref_date.date()),
        }
