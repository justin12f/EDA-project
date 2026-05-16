"""Geographic distribution analysis: frequency by region, country, and city."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeoFrequencyRecord:
    """Immutable frequency record for a single geographic unit."""

    geo_unit: str
    count: int
    proportion: float
    cumulative_proportion: float
    rank: int


class GeoUnitFrequencyCalculator:
    """Computes frequency distribution across geographic units.

    Works for any granularity: country, region, state, city, postal code.
    """

    def calculate(
        self,
        series: pd.Series,
        top_n: int | None,
    ) -> list[GeoFrequencyRecord]:
        """Compute ranked frequency distribution.

        Args:
            series: Series of geographic labels (country, city, etc.).
            top_n: Limit results to top N units. None = all.

        Returns:
            List of GeoFrequencyRecord sorted descending by count.
        """
        value_counts = series.value_counts()
        total = int(value_counts.sum())
        records: list[GeoFrequencyRecord] = []
        cumulative = 0.0

        for rank, (unit, count) in enumerate(value_counts.items(), start=1):
            proportion = count / total
            cumulative += proportion
            records.append(
                GeoFrequencyRecord(
                    geo_unit=str(unit),
                    count=int(count),
                    proportion=round(proportion, 6),
                    cumulative_proportion=round(cumulative, 6),
                    rank=rank,
                )
            )

        return records[:top_n] if top_n is not None else records


class GeoConcentrationCalculator:
    """Computes Herfindahl-Hirschman Index (HHI) for geographic concentration.

    HHI = Σ pᵢ² where pᵢ = share of records in region i.
    HHI → 0: perfectly dispersed across many regions.
    HHI = 1: all records in one region.

    Normalized HHI = (HHI - 1/n) / (1 - 1/n) scales to [0, 1]
    regardless of the number of regions.
    """

    def calculate(self, proportions: np.ndarray) -> dict:
        """Compute HHI and normalized HHI.

        Args:
            proportions: Array of geographic unit proportions (sum to 1).

        Returns:
            Dict with hhi, normalized_hhi, and concentration label.
        """
        n = len(proportions)
        hhi = float(np.sum(proportions ** 2))
        hhi_min = 1.0 / n if n > 0 else 0.0
        normalized = (hhi - hhi_min) / (1.0 - hhi_min) if (1.0 - hhi_min) > 0 else 0.0

        if normalized > 0.6:
            label = "highly_concentrated"
        elif normalized > 0.3:
            label = "moderately_concentrated"
        else:
            label = "dispersed"

        return {
            "hhi": round(hhi, 6),
            "normalized_hhi": round(normalized, 6),
            "concentration_label": label,
            "n_regions": n,
        }


class GeoDistributionCalculator:
    """Geographic frequency distribution with concentration metrics.

    Workflow:
        calculator = GeoDistributionCalculator()
        result = calculator.calculate(
            data_frame=df,
            geo_column="country",
            top_n=20,
            secondary_column="city",   # optional second-level breakdown
        )
    """

    _MINIMUM_RECORDS: int = 1

    def __init__(self) -> None:
        self._freq_calculator = GeoUnitFrequencyCalculator()
        self._concentration_calculator = GeoConcentrationCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        geo_column: str,
        top_n: int | None = 20,
        secondary_column: str | None = None,
    ) -> dict:
        """Compute geographic frequency distribution.

        Args:
            data_frame: Source DataFrame.
            geo_column: Primary geographic label column (country, region, city).
            top_n: Limit primary results to top N. None = all.
            secondary_column: Optional second-level breakdown (e.g., city within country).

        Returns:
            Dict with ranked distribution, concentration metrics, and optional breakdown.

        Raises:
            KeyError: If columns are not found.
            ValueError: If data is insufficient.
        """
        if geo_column not in data_frame.columns:
            raise KeyError(f"Column '{geo_column}' not found in DataFrame.")
        if secondary_column is not None and secondary_column not in data_frame.columns:
            raise KeyError(f"Column '{secondary_column}' not found in DataFrame.")

        clean = data_frame[[geo_column] + ([secondary_column] if secondary_column else [])].dropna(
            subset=[geo_column]
        )

        if len(clean) < self._MINIMUM_RECORDS:
            raise ValueError("No valid records found after dropping null values.")

        records = self._freq_calculator.calculate(clean[geo_column], top_n)
        proportions = np.array([r.proportion for r in records])
        concentration = self._concentration_calculator.calculate(proportions)

        secondary_breakdown: dict[str, list[dict]] | None = None
        if secondary_column is not None:
            top_units = {r.geo_unit for r in records}
            secondary_breakdown = {}
            for unit in top_units:
                subset = clean[clean[geo_column] == unit][secondary_column]
                sub_records = self._freq_calculator.calculate(subset, top_n=5)
                secondary_breakdown[unit] = [
                    {
                        "geo_unit": r.geo_unit,
                        "count": r.count,
                        "proportion": r.proportion,
                        "rank": r.rank,
                    }
                    for r in sub_records
                ]

        return {
            "distribution": [
                {
                    "geo_unit": r.geo_unit,
                    "count": r.count,
                    "proportion": r.proportion,
                    "cumulative_proportion": r.cumulative_proportion,
                    "rank": r.rank,
                }
                for r in records
            ],
            "concentration": concentration,
            "secondary_breakdown": secondary_breakdown,
            "total_records": len(clean),
            "n_unique_units": clean[geo_column].nunique(),
            "geo_column": geo_column,
            "top_n": top_n,
        }
