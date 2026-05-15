"""Financial ratio analysis: profitability, liquidity, leverage, and efficiency."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class RatioCategory(str, Enum):
    """Financial ratio category classification."""

    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"
    EFFICIENCY = "efficiency"


@dataclass(frozen=True)
class FinancialRatio:
    """Immutable computed financial ratio."""

    name: str
    value: float
    category: str
    interpretation: str
    benchmark: str | None


class BaseRatioCalculator(ABC):
    """Abstract base for all financial ratio calculators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Ratio name."""

    @property
    @abstractmethod
    def category(self) -> RatioCategory:
        """Ratio category."""

    @abstractmethod
    def compute(self, row: pd.Series) -> FinancialRatio | None:
        """Compute ratio from a named Series of financial values.

        Args:
            row: Series with financial line items as index.

        Returns:
            FinancialRatio or None if required inputs are unavailable.
        """

    def _get(self, row: pd.Series, key: str) -> float | None:
        """Safe value extractor — returns None if key missing or NaN.

        Args:
            row: Financial data Series.
            key: Field name to extract.

        Returns:
            Float value or None.
        """
        val = row.get(key)
        if val is None or pd.isna(val):
            return None
        return float(val)


class GrossMarginCalculator(BaseRatioCalculator):
    """Gross Margin = (Revenue - COGS) / Revenue.

    Measures production efficiency and pricing power.
    Benchmark: >40% is considered healthy in most industries.
    """

    @property
    def name(self) -> str:
        return "gross_margin"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.PROFITABILITY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        revenue = self._get(row, "revenue")
        cogs = self._get(row, "cogs")
        if revenue is None or cogs is None or revenue == 0:
            return None
        value = (revenue - cogs) / revenue
        return FinancialRatio(
            name=self.name,
            value=round(value, 6),
            category=self.category.value,
            interpretation=f"Gross margin of {value*100:.2f}%. "
                           + ("Strong." if value > 0.4 else "Below typical threshold of 40%."),
            benchmark=">40%",
        )


class NetMarginCalculator(BaseRatioCalculator):
    """Net Profit Margin = Net Income / Revenue.

    Measures overall profitability after all expenses and taxes.
    """

    @property
    def name(self) -> str:
        return "net_margin"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.PROFITABILITY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        revenue = self._get(row, "revenue")
        net_income = self._get(row, "net_income")
        if revenue is None or net_income is None or revenue == 0:
            return None
        value = net_income / revenue
        return FinancialRatio(
            name=self.name,
            value=round(value, 6),
            category=self.category.value,
            interpretation=f"Net margin of {value*100:.2f}%. "
                           + ("Profitable." if value > 0 else "Operating at a loss."),
            benchmark=">10% for most industries",
        )


class ROECalculator(BaseRatioCalculator):
    """Return on Equity = Net Income / Shareholders' Equity.

    Measures how efficiently equity capital generates profits.
    """

    @property
    def name(self) -> str:
        return "roe"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.PROFITABILITY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        net_income = self._get(row, "net_income")
        equity = self._get(row, "shareholders_equity")
        if net_income is None or equity is None or equity == 0:
            return None
        value = net_income / equity
        return FinancialRatio(
            name=self.name,
            value=round(value, 6),
            category=self.category.value,
            interpretation=f"ROE of {value*100:.2f}%. "
                           + ("Strong." if value > 0.15 else "Below 15% benchmark."),
            benchmark=">15%",
        )


class ROACalculator(BaseRatioCalculator):
    """Return on Assets = Net Income / Total Assets.

    Measures how efficiently assets generate profit.
    """

    @property
    def name(self) -> str:
        return "roa"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.PROFITABILITY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        net_income = self._get(row, "net_income")
        total_assets = self._get(row, "total_assets")
        if net_income is None or total_assets is None or total_assets == 0:
            return None
        value = net_income / total_assets
        return FinancialRatio(
            name=self.name,
            value=round(value, 6),
            category=self.category.value,
            interpretation=f"ROA of {value*100:.2f}%. "
                           + ("Efficient asset use." if value > 0.05 else "Below 5% benchmark."),
            benchmark=">5%",
        )


class CurrentRatioCalculator(BaseRatioCalculator):
    """Current Ratio = Current Assets / Current Liabilities.

    Measures short-term liquidity. Ratio < 1 signals potential solvency risk.
    """

    @property
    def name(self) -> str:
        return "current_ratio"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.LIQUIDITY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        current_assets = self._get(row, "current_assets")
        current_liabilities = self._get(row, "current_liabilities")
        if current_assets is None or current_liabilities is None or current_liabilities == 0:
            return None
        value = current_assets / current_liabilities
        return FinancialRatio(
            name=self.name,
            value=round(value, 4),
            category=self.category.value,
            interpretation=f"Current ratio of {value:.2f}. "
                           + ("Adequate liquidity." if value >= 1.5
                              else "Low liquidity — potential repayment risk." if value < 1
                              else "Marginal liquidity."),
            benchmark="1.5 – 3.0",
        )


class QuickRatioCalculator(BaseRatioCalculator):
    """Quick Ratio = (Current Assets - Inventory) / Current Liabilities.

    More conservative than current ratio — excludes inventory (less liquid).
    """

    @property
    def name(self) -> str:
        return "quick_ratio"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.LIQUIDITY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        current_assets = self._get(row, "current_assets")
        inventory = self._get(row, "inventory") or 0.0
        current_liabilities = self._get(row, "current_liabilities")
        if current_assets is None or current_liabilities is None or current_liabilities == 0:
            return None
        value = (current_assets - inventory) / current_liabilities
        return FinancialRatio(
            name=self.name,
            value=round(value, 4),
            category=self.category.value,
            interpretation=f"Quick ratio of {value:.2f}. "
                           + ("Strong liquidity." if value >= 1.0
                              else "May struggle to meet short-term obligations."),
            benchmark=">1.0",
        )


class DebtToEquityCalculator(BaseRatioCalculator):
    """Debt-to-Equity = Total Debt / Shareholders' Equity.

    Measures financial leverage. High D/E = higher risk but potential amplified returns.
    """

    @property
    def name(self) -> str:
        return "debt_to_equity"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.LEVERAGE

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        total_debt = self._get(row, "total_debt")
        equity = self._get(row, "shareholders_equity")
        if total_debt is None or equity is None or equity == 0:
            return None
        value = total_debt / equity
        return FinancialRatio(
            name=self.name,
            value=round(value, 4),
            category=self.category.value,
            interpretation=f"D/E of {value:.2f}. "
                           + ("Conservative leverage." if value < 1.0
                              else "Moderate leverage." if value < 2.0
                              else "High leverage — elevated risk."),
            benchmark="<2.0",
        )


class AssetTurnoverCalculator(BaseRatioCalculator):
    """Asset Turnover = Revenue / Total Assets.

    Measures how efficiently assets generate revenue.
    Capital-intensive industries (utilities) naturally have lower ratios.
    """

    @property
    def name(self) -> str:
        return "asset_turnover"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.EFFICIENCY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        revenue = self._get(row, "revenue")
        total_assets = self._get(row, "total_assets")
        if revenue is None or total_assets is None or total_assets == 0:
            return None
        value = revenue / total_assets
        return FinancialRatio(
            name=self.name,
            value=round(value, 4),
            category=self.category.value,
            interpretation=f"Asset turnover of {value:.2f}x. "
                           + ("Efficient asset utilization." if value > 1.0
                              else "Low asset utilization."),
            benchmark=">1.0x for non-capital-intensive industries",
        )


class InventoryTurnoverCalculator(BaseRatioCalculator):
    """Inventory Turnover = COGS / Average Inventory.

    Measures how frequently inventory is sold and replaced.
    Higher = faster-moving inventory, less obsolescence risk.
    """

    @property
    def name(self) -> str:
        return "inventory_turnover"

    @property
    def category(self) -> RatioCategory:
        return RatioCategory.EFFICIENCY

    def compute(self, row: pd.Series) -> FinancialRatio | None:
        cogs = self._get(row, "cogs")
        inventory = self._get(row, "inventory")
        if cogs is None or inventory is None or inventory == 0:
            return None
        value = cogs / inventory
        return FinancialRatio(
            name=self.name,
            value=round(value, 4),
            category=self.category.value,
            interpretation=f"Inventory turnover of {value:.2f}x. "
                           + ("Fast-moving inventory." if value > 6
                              else "Slow-moving inventory — potential obsolescence risk."),
            benchmark=">6x for retail; industry-dependent",
        )


_RATIO_REGISTRY: dict[str, BaseRatioCalculator] = {
    "gross_margin": GrossMarginCalculator(),
    "net_margin": NetMarginCalculator(),
    "roe": ROECalculator(),
    "roa": ROACalculator(),
    "current_ratio": CurrentRatioCalculator(),
    "quick_ratio": QuickRatioCalculator(),
    "debt_to_equity": DebtToEquityCalculator(),
    "asset_turnover": AssetTurnoverCalculator(),
    "inventory_turnover": InventoryTurnoverCalculator(),
}


class FinancialRatiosCalculator:
    """Configurable financial ratio suite from a named financial data Series/row.

    Workflow:
        calculator = FinancialRatiosCalculator()
        result = calculator.calculate(
            data_frame=df,
            ratios=None,   # optional, defaults to all registered ratios
        )

    Required columns (subset used per ratio):
        revenue, cogs, net_income, shareholders_equity,
        total_assets, current_assets, current_liabilities,
        inventory, total_debt
    """

    def calculate(
        self,
        data_frame: pd.DataFrame,
        ratios: list[str] | None = None,
    ) -> dict:
        """Compute financial ratios for each row in the DataFrame.

        Args:
            data_frame: DataFrame where each row is a period/entity snapshot.
            ratios: Subset of ratios to compute. Defaults to all registered.

        Returns:
            Dict with per-row ratio results and category summary.

        Raises:
            KeyError: If requested ratio keys are not registered.
        """
        active_ratios = ratios if ratios is not None else list(_RATIO_REGISTRY.keys())
        invalid = [r for r in active_ratios if r not in _RATIO_REGISTRY]
        if invalid:
            raise KeyError(
                f"Unknown ratio(s): {invalid}. "
                f"Available: {list(_RATIO_REGISTRY.keys())}"
            )

        rows_results: list[dict] = []

        for idx, row in data_frame.iterrows():
            row_ratios: dict[str, dict] = {}
            for ratio_key in active_ratios:
                calculator = _RATIO_REGISTRY[ratio_key]
                result = calculator.compute(row)
                if result is not None:
                    row_ratios[ratio_key] = {
                        "value": result.value,
                        "category": result.category,
                        "interpretation": result.interpretation,
                        "benchmark": result.benchmark,
                    }
            rows_results.append({
                "row_index": str(idx),
                "ratios": row_ratios,
            })

        return {
            "results": rows_results,
            "ratios_computed": active_ratios,
            "available_ratios": list(_RATIO_REGISTRY.keys()),
            "categories": {cat.value: [] for cat in RatioCategory},
            "n_rows": len(rows_results),
        }
