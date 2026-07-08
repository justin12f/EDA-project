"""PySpark-native backend implementations for the business statistics domain."""
from __future__ import annotations

import math
from typing import Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np

from business.abstract.churn_rate import AbstractChurnRateCalculator
from business.abstract.conversion_funnel import AbstractConversionFunnelCalculator
from business.abstract.customer_lifetime_value import AbstractCustomerLifetimeValueCalculator
from business.abstract.financial_ratios import AbstractFinancialRatiosCalculator
from business.abstract.growth_rates import AbstractGrowthRatesCalculator
from business.abstract.pareto_analysis import AbstractParetoAnalysisCalculator
from business.abstract.risk_metrics import AbstractRiskMetricsCalculator
from business.abstract.run_rate import AbstractRunRateCalculator


class ChurnRateCalculatorSpark(AbstractChurnRateCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        customer_id_column: str,
        start_date_column: str,
        end_date_column: str,
        analysis_start: str,
        analysis_end: str,
    ) -> dict[str, Any]:
        
        start_dt = F.to_date(F.lit(analysis_start))
        end_dt = F.to_date(F.lit(analysis_end))

        # Expressions for conditions
        active_start_expr = (F.col(start_date_column) <= start_dt) & (
            F.col(end_date_column).isNull() | (F.col(end_date_column) > start_dt)
        )
        churned_expr = (F.col(end_date_column) > start_dt) & (F.col(end_date_column) <= end_dt)
        active_end_expr = (F.col(start_date_column) <= end_dt) & (
            F.col(end_date_column).isNull() | (F.col(end_date_column) > end_dt)
        )

        res = data.agg(
            F.countDistinct(F.when(active_start_expr, F.col(customer_id_column))).alias("start"),
            F.countDistinct(F.when(churned_expr, F.col(customer_id_column))).alias("churned"),
            F.countDistinct(F.when(active_end_expr, F.col(customer_id_column))).alias("end")
        ).collect()[0]

        cust_start = int(res["start"])
        cust_churn = int(res["churned"])
        cust_end = int(res["end"])

        avg_cust = (cust_start + cust_end) / 2.0
        churn_rate = cust_churn / cust_start if cust_start > 0 else 0.0

        return {
            "churn_rate": float(churn_rate),
            "customers_start": cust_start,
            "customers_end": cust_end,
            "customers_churned": cust_churn,
            "average_customers": float(avg_cust),
        }


class ConversionFunnelCalculatorSpark(AbstractConversionFunnelCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        step_column: str,
        user_column: str,
        steps_order: list[str],
    ) -> dict[str, Any]:
        counts = data.groupBy(step_column).agg(F.countDistinct(user_column).alias("cnt")).collect()
        count_map = {r[step_column]: r["cnt"] for r in counts}

        funnel = []
        prev_count = None
        first_count = count_map.get(steps_order[0], 0)

        for step in steps_order:
            current_count = count_map.get(step, 0)
            funnel.append({
                "step": step,
                "users": current_count,
                "conversion_from_prev": float(current_count / prev_count) if prev_count else 1.0,
                "conversion_from_start": float(current_count / first_count) if first_count else 0.0,
            })
            prev_count = current_count

        return {
            "funnel": funnel,
            "overall_conversion": funnel[-1]["conversion_from_start"] if funnel else 0.0,
        }


class CustomerLifetimeValueCalculatorSpark(AbstractCustomerLifetimeValueCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        customer_column: str,
        order_value_column: str,
        date_column: str,
        discount_rate: float = 0.1,
        margin_rate: float = 0.3,
        periods_per_year: int = 12,
    ) -> dict[str, Any]:
        
        metrics = data.groupBy(customer_column).agg(
            F.mean(order_value_column).alias("avg_order_value"),
            F.count("*").alias("total_orders"),
            F.min(date_column).alias("first_date"),
            F.max(date_column).alias("last_date")
        ).withColumn(
            "lifespan_days", F.datediff(F.col("last_date"), F.col("first_date"))
        )

        max_lifespan_days = float(metrics.agg(F.max("lifespan_days")).collect()[0][0] or 0)
        observation_periods = max(max_lifespan_days / 365.0 * periods_per_year, 1.0)
        
        periodic_discount = discount_rate / periods_per_year

        def calc_dcf(aov: float, freq: float, lifespan: float) -> float:
            margin = aov * freq * margin_rate
            n = int(lifespan)
            dfactor = 1.0 / (1.0 + periodic_discount)
            pv_factor = (1 - dfactor**n) / (1 - dfactor) if dfactor != 1 else n
            return margin * pv_factor

        records = []
        for row in metrics.collect():
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


class FinancialRatiosCalculatorSpark(AbstractFinancialRatiosCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        revenue_column: str,
        cost_column: str,
        equity_column: str | None = None,
        assets_column: str | None = None,
    ) -> dict[str, Any]:
        
        agg_exprs = [
            F.sum(revenue_column).alias("rev"),
            F.sum(cost_column).alias("cost")
        ]
        if equity_column:
            agg_exprs.append(F.sum(equity_column).alias("equity"))
        if assets_column:
            agg_exprs.append(F.sum(assets_column).alias("assets"))

        res = data.agg(*agg_exprs).collect()[0]
        
        rev = float(res["rev"] or 0)
        cost = float(res["cost"] or 0)
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
            equity = float(res["equity"] or 0)
            metrics["roe"] = profit / equity if equity > 0 else 0.0
        if assets_column:
            assets = float(res["assets"] or 0)
            metrics["roa"] = profit / assets if assets > 0 else 0.0

        return metrics


class GrowthRatesCalculatorSpark(AbstractGrowthRatesCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        date_column: str,
        value_column: str,
        periods: int = 1,
    ) -> dict[str, Any]:
        
        window = Window.orderBy(date_column)
        df = data.withColumn("prev_val", F.lag(value_column, periods).over(window))
        df = df.withColumn("growth", (F.col(value_column) - F.col("prev_val")) / F.col("prev_val"))
        
        rows = df.select(date_column, value_column, "growth").orderBy(date_column).collect()
        res = [
            {date_column: r[date_column], value_column: r[value_column], "growth": r["growth"]}
            for r in rows
        ]
        
        first = float(rows[0][value_column]) if rows else 0.0
        last = float(rows[-1][value_column]) if rows else 0.0
        n = len(rows)

        cagr = (last / first)**(1.0 / n) - 1 if first > 0 and n > 0 else 0.0

        return {
            "period_growth": res,
            "cagr": float(cagr),
            "first_value": first,
            "last_value": last,
            "n_periods": n,
        }


class ParetoAnalysisCalculatorSpark(AbstractParetoAnalysisCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        entity_column: str,
        value_column: str,
    ) -> dict[str, Any]:
        
        agg_df = data.groupBy(entity_column).agg(F.sum(value_column).alias("val"))
        total_val = float(agg_df.agg(F.sum("val")).collect()[0][0] or 0.0)
        
        window = Window.orderBy(F.desc("val"))
        
        df = agg_df.withColumn("cum_val", F.sum("val").over(window))
        df = df.withColumn("pct", F.col("val") / total_val)
        df = df.withColumn("cum_pct", F.col("cum_val") / total_val)
        
        def assign_seg(c: float) -> str:
            if c <= 0.8: return "A (Top 80%)"
            elif c <= 0.95: return "B (Next 15%)"
            else: return "C (Bottom 5%)"

        rows = df.orderBy(F.desc("val")).collect()
        res = []
        for r in rows:
            res.append({
                "entity": r[entity_column],
                "value": float(r["val"]),
                "cumulative_percentage": float(r["cum_pct"]),
                "segment": assign_seg(float(r["cum_pct"]))
            })

        return {"pareto_table": res}


class RiskMetricsCalculatorSpark(AbstractRiskMetricsCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        returns_column: str,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        
        clean = data.select(returns_column).dropna()
        n = clean.count()

        agg = clean.agg(
            F.mean(returns_column).alias("mean"),
            F.stddev_samp(returns_column).alias("std"),
            F.percentile_approx(returns_column, 1 - confidence_level).alias("var")
        ).collect()[0]

        mean_ret = float(agg["mean"] or 0.0)
        std_ret = float(agg["std"] or 0.0)
        var = float(agg["var"] or 0.0)
        
        # Sortino
        downside_df = clean.filter(F.col(returns_column) < 0)
        downside_sq_sum = downside_df.withColumn("sq", F.col(returns_column)**2).agg(F.sum("sq")).collect()[0][0]
        down_std = math.sqrt(float(downside_sq_sum or 0.0) / n) if n > 0 else 0.0

        sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0.0
        sortino = (mean_ret - risk_free_rate) / down_std if down_std > 0 else 0.0

        # CVaR
        cvar_rets = clean.filter(F.col(returns_column) <= var)
        cvar = float(cvar_rets.agg(F.mean(returns_column)).collect()[0][0] or 0.0)

        return {
            "mean_return": mean_ret,
            "volatility": std_ret,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "value_at_risk": var,
            "conditional_value_at_risk": cvar,
        }


class RunRateCalculatorSpark(AbstractRunRateCalculator):
    def calculate(
        self,
        data: SparkDataFrame,
        date_column: str,
        value_column: str,
        extrapolation_periods: int = 12,
    ) -> dict[str, Any]:
        
        agg = data.agg(
            F.sum(value_column).alias("total"),
            F.count("*").alias("n")
        ).collect()[0]
        
        total = float(agg["total"] or 0.0)
        n = int(agg["n"] or 0)

        avg_per_period = total / n if n > 0 else 0.0
        run_rate = avg_per_period * extrapolation_periods

        return {
            "current_total": total,
            "periods_observed": n,
            "average_per_period": avg_per_period,
            "projected_run_rate": run_rate,
            "extrapolation_multiplier": extrapolation_periods
        }
