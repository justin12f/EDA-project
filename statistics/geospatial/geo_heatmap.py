"""Geographic density heatmap via grid-cell aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HeatmapCell:
    """Immutable heatmap grid cell."""

    row: int
    col: int
    lat_center: float
    lon_center: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    count: int
    density: float
    normalized_density: float


class GridBoundaryComputer:
    """Computes grid boundaries and cell dimensions from coordinate extents."""

    def compute(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        n_lat_bins: int,
        n_lon_bins: int,
        padding_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute lat/lon bin edges with optional padding.

        Args:
            lat: Latitude array.
            lon: Longitude array.
            n_lat_bins: Number of latitude bins.
            n_lon_bins: Number of longitude bins.
            padding_factor: Fractional padding added to range extents.

        Returns:
            Tuple (lat_edges, lon_edges) as 1D bin edge arrays.
        """
        lat_range = float(lat.max() - lat.min())
        lon_range = float(lon.max() - lon.min())

        lat_pad = lat_range * padding_factor if lat_range > 0 else 0.1
        lon_pad = lon_range * padding_factor if lon_range > 0 else 0.1

        lat_edges = np.linspace(
            float(lat.min()) - lat_pad,
            float(lat.max()) + lat_pad,
            n_lat_bins + 1,
        )
        lon_edges = np.linspace(
            float(lon.min()) - lon_pad,
            float(lon.max()) + lon_pad,
            n_lon_bins + 1,
        )

        return lat_edges, lon_edges


class GridCountAccumulator:
    """Bins coordinate points into a 2D count grid using numpy histogram2d."""

    def accumulate(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        lat_edges: np.ndarray,
        lon_edges: np.ndarray,
    ) -> np.ndarray:
        """Accumulate point counts per grid cell.

        Args:
            lat: Latitude array.
            lon: Longitude array.
            lat_edges: Latitude bin edges.
            lon_edges: Longitude bin edges.

        Returns:
            2D count matrix of shape (n_lat_bins, n_lon_bins).
        """
        counts, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
        return counts


class CellDensityCalculator:
    """Converts raw counts to density (points per km²).

    Cell area is approximated using the equirectangular projection:
        area ≈ (Δlat_km) × (Δlon_km)
    where Δlon_km varies with latitude.
    """

    _EARTH_RADIUS_KM: float = 6371.0
    _DEG_TO_RAD: float = np.pi / 180

    def calculate(
        self,
        counts: np.ndarray,
        lat_edges: np.ndarray,
        lon_edges: np.ndarray,
    ) -> np.ndarray:
        """Compute density matrix in points per km².

        Args:
            counts: 2D count matrix.
            lat_edges: Latitude bin edges.
            lon_edges: Longitude bin edges.

        Returns:
            2D density matrix (same shape as counts).
        """
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        delta_lat_km = (lat_edges[1] - lat_edges[0]) * self._EARTH_RADIUS_KM * self._DEG_TO_RAD
        delta_lon_deg = lon_edges[1] - lon_edges[0]

        density = np.zeros_like(counts)
        for i, lat_c in enumerate(lat_centers):
            delta_lon_km = (
                delta_lon_deg * self._EARTH_RADIUS_KM * self._DEG_TO_RAD * np.cos(lat_c * self._DEG_TO_RAD)
            )
            cell_area_km2 = delta_lat_km * delta_lon_km
            if cell_area_km2 > 0:
                density[i, :] = counts[i, :] / cell_area_km2

        return density


class GeoHeatmapCalculator:
    """Grid-based geographic density heatmap.

    Divides the coordinate bounding box into a regular grid and
    counts points per cell. Outputs both raw counts and
    density (points per km²) for scale-independent comparison.

    Workflow:
        calculator = GeoHeatmapCalculator()
        result = calculator.calculate(
            data_frame=df,
            lat_column="latitude",
            lon_column="longitude",
            n_lat_bins=20,
            n_lon_bins=20,
            include_empty_cells=False,
        )
    """

    _MINIMUM_POINTS: int = 5
    _DEFAULT_PADDING: float = 0.05

    def __init__(self) -> None:
        self._grid_boundary = GridBoundaryComputer()
        self._count_accumulator = GridCountAccumulator()
        self._density_calculator = CellDensityCalculator()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        lat_column: str,
        lon_column: str,
        n_lat_bins: int = 20,
        n_lon_bins: int = 20,
        include_empty_cells: bool = False,
    ) -> dict:
        """Build geographic density heatmap.

        Args:
            data_frame: Source DataFrame.
            lat_column: Latitude column (decimal degrees).
            lon_column: Longitude column (decimal degrees).
            n_lat_bins: Number of latitude grid divisions.
            n_lon_bins: Number of longitude grid divisions.
            include_empty_cells: Whether to include cells with zero points.

        Returns:
            Dict with heatmap grid cells, density statistics, and peak cell.

        Raises:
            KeyError: If coordinate columns are not found.
            ValueError: If data is insufficient or bin counts are invalid.
        """
        for col in (lat_column, lon_column):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        if n_lat_bins < 2:
            raise ValueError(f"n_lat_bins must be >= 2. Got {n_lat_bins}.")
        if n_lon_bins < 2:
            raise ValueError(f"n_lon_bins must be >= 2. Got {n_lon_bins}.")

        clean = data_frame[[lat_column, lon_column]].dropna()

        if len(clean) < self._MINIMUM_POINTS:
            raise ValueError(
                f"At least {self._MINIMUM_POINTS} coordinate pairs required. "
                f"Got {len(clean)}."
            )

        lat = clean[lat_column].to_numpy(dtype=float)
        lon = clean[lon_column].to_numpy(dtype=float)

        lat_edges, lon_edges = self._grid_boundary.compute(
            lat, lon, n_lat_bins, n_lon_bins, self._DEFAULT_PADDING
        )
        counts = self._count_accumulator.accumulate(lat, lon, lat_edges, lon_edges)
        density = self._density_calculator.calculate(counts, lat_edges, lon_edges)

        max_density = float(density.max())
        normalized = density / max_density if max_density > 0 else density

        cells: list[HeatmapCell] = []
        peak_cell: HeatmapCell | None = None
        peak_density = -1.0

        for i in range(n_lat_bins):
            for j in range(n_lon_bins):
                count = int(counts[i, j])
                if not include_empty_cells and count == 0:
                    continue

                cell = HeatmapCell(
                    row=i,
                    col=j,
                    lat_center=round(float((lat_edges[i] + lat_edges[i + 1]) / 2), 6),
                    lon_center=round(float((lon_edges[j] + lon_edges[j + 1]) / 2), 6),
                    lat_min=round(float(lat_edges[i]), 6),
                    lat_max=round(float(lat_edges[i + 1]), 6),
                    lon_min=round(float(lon_edges[j]), 6),
                    lon_max=round(float(lon_edges[j + 1]), 6),
                    count=count,
                    density=round(float(density[i, j]), 6),
                    normalized_density=round(float(normalized[i, j]), 6),
                )
                cells.append(cell)

                if float(density[i, j]) > peak_density:
                    peak_density = float(density[i, j])
                    peak_cell = cell

        non_empty_counts = counts[counts > 0]

        return {
            "cells": [
                {
                    "row": c.row, "col": c.col,
                    "lat_center": c.lat_center, "lon_center": c.lon_center,
                    "lat_min": c.lat_min, "lat_max": c.lat_max,
                    "lon_min": c.lon_min, "lon_max": c.lon_max,
                    "count": c.count,
                    "density_per_km2": c.density,
                    "normalized_density": c.normalized_density,
                }
                for c in cells
            ],
            "peak_cell": {
                "lat_center": peak_cell.lat_center,
                "lon_center": peak_cell.lon_center,
                "count": peak_cell.count,
                "density_per_km2": peak_cell.density,
            } if peak_cell else None,
            "grid_summary": {
                "n_lat_bins": n_lat_bins,
                "n_lon_bins": n_lon_bins,
                "total_cells": n_lat_bins * n_lon_bins,
                "occupied_cells": int((counts > 0).sum()),
                "occupation_ratio": round(float((counts > 0).mean()), 4),
                "mean_count_per_occupied_cell": round(float(non_empty_counts.mean()), 2) if len(non_empty_counts) > 0 else 0.0,
                "max_density_per_km2": round(max_density, 4),
            },
            "n_points": len(clean),
        }
