"""Customer Lifetime Value (CLV) — simple and probabilistic estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CLVRecord:
    """Immutable CLV record for a single customer."""

    entity_id: str
    avg_order_value: float
    purchase_frequency: float
    customer_lifespan: float
    simple_clv: float
    discounted_clv: float
    segment: str


class SimpleCLVCalculator:
    """Simple (non-probabilistic) CLV formula.

    CLV = avg_order_value × purchase_frequency × customer_lifespan

    All inputs must share the same time unit (e.g., annual).
    """

    def calculate(
        self,
        avg_order_value: float,
        purchase_frequency: float,
        customer_lifespan: float,
    ) -> float:
        """Compute simple CLV.

        Args:
            avg_order_value: Average revenue per transaction.
            purchase_frequency: Expected transactions per period.
            customer_lifespan: Expected customer lifespan in periods.

        Returns:
            Simple CLV value.
        """
        return avg_order_value * purchase_frequency * customer_lifespan


class DiscountedCLVCalculator:
    """Discounted CLV using geometric series present value formula.

    DCF-CLV = Σₜ₌₁ⁿ [margin × (1 / (1+d)^t)]
            = margin × [(1-(1/(1+d))^n) / (1 - 1/(1+d))]

    where margin = avg_order_value × purchase_frequency per period,
    d = discount rate, n = lifespan in periods.
    """

    def calculate(
        self,
        avg_order_value: float,
        purchase_frequency: float,
        customer_lifespan: float,
        discount_rate: float,
        margin_rate: float,
    ) -> float:
        """Compute discounted CLV.

        Args:
            avg_order_value: Average revenue per transaction.
            purchase_frequency: Transactions per period.
            customer_lifespan: Lifespan in periods.
            discount_rate: Periodic discount rate (e.g., 0.1 = 10% per period).
            margin_rate: Profit margin as decimal.

        Returns:
            Discounted CLV (present value of future margins).
        """
        if discount_rate <= 0:
            # No discounting — fall back to simple CLV with margin
            return avg_order_value * purchase_frequency * customer_lifespan * margin_rate

        margin_per_period = avg_order_value * purchase_frequency * margin_rate
        n = int(customer_lifespan)
        discount_factor = 1.0 / (1.0 + discount_rate)
        pv_factor = (1 - discount_factor ** n) / (1 - discount_factor) if discount_factor != 1 else n
        return margin_per_period * pv_factor


class CLVSegmentAssigner:
    """Assigns CLV-based segment label using percentile thresholds.

    Segments:
        Top 20% by CLV    → 'high_value'
        Middle 60%        → 'mid_value'
        Bottom 20%        → 'low_value'
    """

    def assign(self, clv_values: np.ndarray) -> list[str]:
        """Assign segment labels based on CLV percentile.

        Args:
            clv_values: Array of CLV values.

        Returns:
            List of segment label strings.
        """
        p80 = float(np.percentile(clv_values, 80))
        p20 = float(np.percentile(clv_values, 20))

        return [
            "high_value" if v >= p80
            else "low_value" if v <= p20
            else "mid_value"
            for v in clv_values
        ]


class CustomerLifetimeValueCalculator:
    """CLV computation from transactional data per customer.

    Workflow:
        calculator = CustomerLifetimeValueCalculator()
        result = calculator.calculate(
            data_frame=df,
            customer_column="customer_id",
            order_value_column="order_value",
            date_column="purchase_date",
            discount_rate=0.1,
            margin_rate=0.3,
            periods_per_year=12,
        )
    """

    _MINIMUM_CUSTOMERS: int = 5

    def __init__(self) -> None:
        self._simple_calc = SimpleCLVCalculator()
        self._discounted_calc = DiscountedCLVCalculator()
        self._segment_assigner = CLVSegmentAssigner()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        customer_column: str,
        order_value_column: str,
        date_column: str,
        discount_rate: float = 0.1,
        margin_rate: float = 0.3,
        periods_per_year: int = 12,
    ) -> dict:
        """Compute CLV per customer from transactional data.

        Args:
            data_frame: Transactional DataFrame.
            customer_column: Customer identifier column.
            order_value_column: Transaction amount column.
            date_column: Transaction date column.
            discount_rate: Annual discount rate as decimal.
            margin_rate: Gross margin as decimal.
            periods_per_year: Periods per year (12=monthly frequency).

        Returns:
            Dict with per-customer CLV, segments, and portfolio summary.

        Raises:
            KeyError: If required columns are not found.
            ValueError: If parameters are invalid or data is insufficient.
        """
        required = [customer_column, order_value_column, date_column]
        missing = [c for c in required if c not in data_frame.columns]
        if missing:
            raise KeyError(f"Required columns not found: {missing}")
        if not 0.0 <= discount_rate:
            raise ValueError(f"discount_rate must be >= 0. Got {discount_rate}.")
        if not 0.0 < margin_rate <= 1.0:
            raise ValueError(f"margin_rate must be in (0, 1]. Got {margin_rate}.")
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1. Got {periods_per_year}.")

        clean = data_frame[required].dropna()
        clean[date_column] = pd.to_datetime(clean[date_column])

        n_customers = clean[customer_column].nunique()
        if n_customers < self._MINIMUM_CUSTOMERS:
            raise ValueError(
                f"At least {self._MINIMUM_CUSTOMERS} unique customers required. "
                f"Got {n_customers}."
            )

        # Compute per-customer metrics
        customer_metrics = (
            clean.groupby(customer_column)
            .agg(
                avg_order_value=(order_value_column, "mean"),
                total_orders=(order_value_column, "count"),
                first_date=(date_column, "min"),
                last_date=(date_column, "max"),
            )
            .reset_index()
        )

        # Lifespan in periods
        customer_metrics["lifespan_days"] = (
            customer_metrics["last_date"] - customer_metrics["first_date"]
        ).dt.days

        observation_window_periods = max(
            float(customer_metrics["lifespan_days"].max() / 365 * periods_per_year),
            1.0,
        )

        customer_metrics["purchase_frequency"] = (
            customer_metrics["total_orders"] / observation_window_periods
        )
        customer_metrics["customer_lifespan"] = (
            customer_metrics["lifespan_days"] / 365 * periods_per_year
        ).clip(lower=1.0)

        periodic_discount = discount_rate / periods_per_year

        records: list[CLVRecord] = []
        for _, row in customer_metrics.iterrows():
            aov = float(row["avg_order_value"])
            freq = float(row["purchase_frequency"])
            lifespan = float(row["customer_lifespan"])

            simple = self._simple_calc.calculate(aov, freq, lifespan)
            discounted = self._discounted_calc.calculate(
                aov, freq, lifespan, periodic_discount, margin_rate
            )

            records.append(
                CLVRecord(
                    entity_id=str(row[customer_column]),
                    avg_order_value=round(aov, 2),
                    purchase_frequency=round(freq, 4),
                    customer_lifespan=round(lifespan, 2),
                    simple_clv=round(simple, 2),
                    discounted_clv=round(discounted, 2),
                    segment="",  # assigned below
                )
            )

        clv_values = np.array([r.discounted_clv for r in records])
        segments = self._segment_assigner.assign(clv_values)

        records = [
            CLVRecord(
                entity_id=r.entity_id,
                avg_order_value=r.avg_order_value,
                purchase_frequency=r.purchase_frequency,
                customer_lifespan=r.customer_lifespan,
                simple_clv=r.simple_clv,
                discounted_clv=r.discounted_clv,
                segment=seg,
            )
            for r, seg in zip(records, segments)
        ]

        segment_counts = {s: segments.count(s) for s in set(segments)}

        return {
            "customers": [
                {
                    "entity_id": r.entity_id,
                    "avg_order_value": r.avg_order_value,
                    "purchase_frequency": r.purchase_frequency,
                    "customer_lifespan_periods": r.customer_lifespan,
                    "simple_clv": r.simple_clv,
                    "discounted_clv": r.discounted_clv,
                    "segment": r.segment,
                }
                for r in records
            ],
            "portfolio_summary": {
                "mean_discounted_clv": round(float(clv_values.mean()), 2),
                "median_discounted_clv": round(float(np.median(clv_values)), 2),
                "total_portfolio_clv": round(float(clv_values.sum()), 2),
                "top_20pct_clv_share": round(
                    float(np.sort(clv_values)[-int(len(clv_values) * 0.2):].sum() / clv_values.sum()),
                    4,
                ) if clv_values.sum() > 0 else 0.0,
                "segment_distribution": segment_counts,
            },
            "parameters": {
                "discount_rate": discount_rate,
                "margin_rate": margin_rate,
                "periods_per_year": periods_per_year,
            },
            "n_customers": len(records),
        }
