"""Polars-native backend implementations for the business statistics domain."""
from __future__ import annotations

from typing import Any
import math

import polars as pl
import numpy as np

from lumen.statistics.business.abstract.churn_rate import AbstractChurnRateCalculator
from lumen.statistics.business.abstract.conversion_funnel import AbstractConversionFunnelCalculator
from lumen.statistics.business.abstract.customer_lifetime_value import AbstractCustomerLifetimeValueCalculator
from lumen.statistics.business.abstract.financial_ratios import AbstractFinancialRatiosCalculator
from lumen.statistics.business.abstract.growth_rates import AbstractGrowthRatesCalculator
from lumen.statistics.business.abstract.pareto_analysis import AbstractParetoAnalysisCalculator
from lumen.statistics.business.abstract.risk_metrics import AbstractRiskMetricsCalculator
from lumen.statistics.business.abstract.run_rate import AbstractRunRateCalculator

def _eager(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


class ChurnRateCalculatorPolars(AbstractChurnRateCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        customer_id_column: str,
        start_date_column: str,
        end_date_column: str,
        analysis_start: str,
        analysis_end: str,
    ) -> dict[str, Any]:
        frame = _eager(data).with_columns([
            pl.col(start_date_column).cast(pl.Date),
            pl.col(end_date_column).cast(pl.Date)
        ])
        start_dt = pl.lit(analysis_start).cast(pl.Date)
        end_dt = pl.lit(analysis_end).cast(pl.Date)

        # Active at start: started before/on start, and ended after start (or never ended)
        active_start_expr = (pl.col(start_date_column) <= start_dt) & (
            pl.col(end_date_column).is_null() | (pl.col(end_date_column) > start_dt)
        )
        # Churned during period: ended between start and end
        churned_expr = (pl.col(end_date_column) > start_dt) & (pl.col(end_date_column) <= end_dt)
        # Active at end
        active_end_expr = (pl.col(start_date_column) <= end_dt) & (
            pl.col(end_date_column).is_null() | (pl.col(end_date_column) > end_dt)
        )

        res = frame.select([
            pl.col(customer_id_column).filter(active_start_expr).n_unique().alias("customers_start"),
            pl.col(customer_id_column).filter(churned_expr).n_unique().alias("customers_churned"),
            pl.col(customer_id_column).filter(active_end_expr).n_unique().alias("customers_end")
        ]).row(0)

        cust_start, cust_churn, cust_end = res
        avg_cust = (cust_start + cust_end) / 2.0
        churn_rate = cust_churn / cust_start if cust_start > 0 else 0.0

        return {
            "churn_rate": float(churn_rate),
            "customers_start": int(cust_start),
            "customers_end": int(cust_end),
            "customers_churned": int(cust_churn),
            "average_customers": float(avg_cust),
        }


class ConversionFunnelCalculatorPolars(AbstractConversionFunnelCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        step_column: str,
        user_column: str,
        steps_order: list[str],
    ) -> dict[str, Any]:
        frame = _eager(data)
        
        counts = frame.group_by(step_column).agg(pl.col(user_column).n_unique().alias("count")).to_dicts()
        count_map = {row[step_column]: row["count"] for row in counts}

        funnel = []
        prev_count = None
        first_count = count_map.get(steps_order[0], 0)

        for step in steps_order:
            current_count = count_map.get(step, 0)
            funnel.append({
                "step": step,
                "users": int(current_count),
                "conversion_from_prev": float(current_count / prev_count) if prev_count else 1.0,
                "conversion_from_start": float(current_count / first_count) if first_count else 0.0,
            })
            prev_count = current_count

        return {
            "funnel": funnel,
            "overall_conversion": funnel[-1]["conversion_from_start"] if funnel else 0.0,
        }


class CustomerLifetimeValueCalculatorPolars(AbstractCustomerLifetimeValueCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        customer_column: str,
        order_value_column: str,
        date_column: str,
        discount_rate: float = 0.1,
        margin_rate: float = 0.3,
        periods_per_year: int = 12,
    ) -> dict[str, Any]:
        frame = _eager(data).with_columns(pl.col(date_column).cast(pl.Date))
        
        metrics = frame.group_by(customer_column).agg([
            pl.col(order_value_column).mean().alias("avg_order_value"),
            pl.len().alias("total_orders"),
            pl.col(date_column).min().alias("first_date"),
            pl.col(date_column).max().alias("last_date")
        ]).with_columns([
            (pl.col("last_date") - pl.col("first_date")).dt.total_days().alias("lifespan_days")
        ])

        max_lifespan_days = float(metrics["lifespan_days"].max() or 0)
        observation_periods = max(max_lifespan_days / 365.0 * periods_per_year, 1.0)
        
        periodic_discount = discount_rate / periods_per_year

        def calc_dcf(aov: float, freq: float, lifespan: float) -> float:
            margin = aov * freq * margin_rate
            n = int(lifespan)
            dfactor = 1.0 / (1.0 + periodic_discount)
            pv_factor = (1 - dfactor**n) / (1 - dfactor) if dfactor != 1 else n
            return margin * pv_factor

        records = []
        for row in metrics.to_dicts():
            aov = row["avg_order_value"]
            freq = row["total_orders"] / observation_periods
            lifespan = max(row["lifespan_days"] / 365.0 * periods_per_year, 1.0)
            
            simple = aov * freq * lifespan * margin_rate
            dcf = calc_dcf(aov, freq, lifespan) if discount_rate > 0 else simple
            
            records.append({
                "entity_id": row[customer_column],
                "avg_order_value": aov,
                "purchase_frequency": freq,
                "customer_lifespan_periods": lifespan,
                "simple_clv": simple,
                "discounted_clv": dcf
            })
            
        clvs = [r["discounted_clv"] for r in records]
        p80 = float(np.percentile(clvs, 80)) if clvs else 0.0
        p20 = float(np.percentile(clvs, 20)) if clvs else 0.0

        for r in records:
            r["segment"] = "high_value" if r["discounted_clv"] >= p80 else "low_value" if r["discounted_clv"] <= p20 else "mid_value"

        return {
            "customers": records,
            "portfolio_summary": {
                "mean_discounted_clv": float(np.mean(clvs)) if clvs else 0.0,
                "total_portfolio_clv": float(np.sum(clvs)) if clvs else 0.0,
            }
        }


class FinancialRatiosCalculatorPolars(AbstractFinancialRatiosCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        revenue_column: str,
        cost_column: str,
        equity_column: str | None = None,
        assets_column: str | None = None,
    ) -> dict[str, Any]:
        frame = _eager(data)
        agg_exprs = [
            pl.col(revenue_column).sum().alias("rev"),
            pl.col(cost_column).sum().alias("cost")
        ]
        if equity_column:
            agg_exprs.append(pl.col(equity_column).sum().alias("equity"))
        if assets_column:
            agg_exprs.append(pl.col(assets_column).sum().alias("assets"))

        res = frame.select(agg_exprs).row(0)
        
        rev = float(res[0])
        cost = float(res[1])
        profit = rev - cost
        margin = profit / rev if rev > 0 else 0.0
        
        metrics: dict[str, Any] = {
            "total_revenue": rev,
            "total_cost": cost,
            "net_profit": profit,
            "profit_margin": margin,
            "roi": profit / cost if cost > 0 else float('inf')
        }

        if equity_column:
            equity = float(res[2])
            metrics["roe"] = profit / equity if equity > 0 else 0.0
        if assets_column:
            assets = float(res[-1])
            metrics["roa"] = profit / assets if assets > 0 else 0.0

        return metrics


class GrowthRatesCalculatorPolars(AbstractGrowthRatesCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        date_column: str,
        value_column: str,
        periods: int = 1,
    ) -> dict[str, Any]:
        frame = _eager(data).sort(date_column)
        
        res = frame.with_columns([
            pl.col(value_column).shift(periods).alias("prev_val")
        ]).with_columns([
            ((pl.col(value_column) - pl.col("prev_val")) / pl.col("prev_val")).alias("growth")
        ]).select([date_column, value_column, "growth"]).to_dicts()

        first = float(frame[value_column].drop_nulls()[0])
        last = float(frame[value_column].drop_nulls()[-1])
        n = frame.height

        cagr = (last / first)**(1.0 / n) - 1 if first > 0 and n > 0 else 0.0

        return {
            "period_growth": res,
            "cagr": float(cagr),
            "first_value": first,
            "last_value": last,
            "n_periods": n,
        }


class ParetoAnalysisCalculatorPolars(AbstractParetoAnalysisCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        entity_column: str,
        value_column: str,
    ) -> dict[str, Any]:
        frame = _eager(data).group_by(entity_column).agg(pl.col(value_column).sum().alias("val"))
        frame = frame.sort("val", descending=True)

        total_val = float(frame["val"].sum())
        
        frame = frame.with_columns([
            (pl.col("val") / total_val).alias("pct"),
            (pl.col("val").cum_sum() / total_val).alias("cum_pct")
        ])

        def assign_seg(c: float) -> str:
            if c <= 0.8: return "A (Top 80%)"
            elif c <= 0.95: return "B (Next 15%)"
            else: return "C (Bottom 5%)"

        res = []
        for row in frame.to_dicts():
            res.append({
                "entity": row[entity_column],
                "value": float(row["val"]),
                "cumulative_percentage": float(row["cum_pct"]),
                "segment": assign_seg(row["cum_pct"])
            })

        return {"pareto_table": res}


class RiskMetricsCalculatorPolars(AbstractRiskMetricsCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        returns_column: str,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        frame = _eager(data).select(pl.col(returns_column).drop_nulls())
        rets = frame[returns_column]
        n = rets.len()

        mean_ret = float(rets.mean() or 0.0)
        std_ret = float(rets.std(ddof=1) or 0.0)
        
        # Sortino needs downside deviation
        downside = rets.filter(rets < 0)
        down_std = float(math.sqrt((downside**2).sum() / n)) if downside.len() > 0 else 0.0

        sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0.0
        sortino = (mean_ret - risk_free_rate) / down_std if down_std > 0 else 0.0

        alpha = 1.0 - confidence_level
        var = float(rets.quantile(alpha, interpolation="linear") or 0.0)

        # CVaR (Expected Shortfall)
        cvar_rets = rets.filter(rets <= var)
        cvar = float(cvar_rets.mean() or 0.0)

        return {
            "mean_return": mean_ret,
            "volatility": std_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "value_at_risk": var,
            "conditional_value_at_risk": cvar,
        }


class RunRateCalculatorPolars(AbstractRunRateCalculator):
    def calculate(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        date_column: str,
        value_column: str,
        extrapolation_periods: int = 12,
    ) -> dict[str, Any]:
        frame = _eager(data).sort(date_column)
        
        total = float(frame[value_column].sum() or 0.0)
        n = frame.height

        avg_per_period = total / n if n > 0 else 0.0
        run_rate = avg_per_period * extrapolation_periods

        return {
            "current_total": total,
            "periods_observed": n,
            "average_per_period": avg_per_period,
            "projected_run_rate": run_rate,
            "extrapolation_multiplier": extrapolation_periods
        }
