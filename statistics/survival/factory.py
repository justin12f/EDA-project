"""Factory — domain `survival`."""
from __future__ import annotations
from typing import Any
from core.abstract_factory import RegistryFactory

class SurvivalStatisticsFactory(RegistryFactory[str, Any]):
    pass

def _register() -> None:
    from statistics.survival.backends import pandas_impl as p
    from statistics.survival.backends import polars_impl as pl
    from statistics.survival.backends import spark_impl as sp

    SurvivalStatisticsFactory.register("interval_extractor", "pandas", p.IntervalExtractorPandas)
    SurvivalStatisticsFactory.register("interval_extractor", "polars", pl.IntervalExtractorPolars)
    SurvivalStatisticsFactory.register("interval_extractor", "spark", sp.IntervalExtractorSpark)
    SurvivalStatisticsFactory.register("regularity_classifier", "pandas", p.RegularityClassifierPandas)
    SurvivalStatisticsFactory.register("regularity_classifier", "polars", pl.RegularityClassifierPolars)
    SurvivalStatisticsFactory.register("regularity_classifier", "spark", sp.RegularityClassifierSpark)
    SurvivalStatisticsFactory.register("temporal_burstiness_calculator", "pandas", p.TemporalBurstinessCalculatorPandas)
    SurvivalStatisticsFactory.register("temporal_burstiness_calculator", "polars", pl.TemporalBurstinessCalculatorPolars)
    SurvivalStatisticsFactory.register("temporal_burstiness_calculator", "spark", sp.TemporalBurstinessCalculatorSpark)
    SurvivalStatisticsFactory.register("rolling_event_rate_calculator", "pandas", p.RollingEventRateCalculatorPandas)
    SurvivalStatisticsFactory.register("rolling_event_rate_calculator", "polars", pl.RollingEventRateCalculatorPolars)
    SurvivalStatisticsFactory.register("rolling_event_rate_calculator", "spark", sp.RollingEventRateCalculatorSpark)
    SurvivalStatisticsFactory.register("event_density_calculator", "pandas", p.EventDensityCalculatorPandas)
    SurvivalStatisticsFactory.register("event_density_calculator", "polars", pl.EventDensityCalculatorPolars)
    SurvivalStatisticsFactory.register("event_density_calculator", "spark", sp.EventDensityCalculatorSpark)
    SurvivalStatisticsFactory.register("nelson_aalen_estimator", "pandas", p.NelsonAalenEstimatorPandas)
    SurvivalStatisticsFactory.register("nelson_aalen_estimator", "polars", pl.NelsonAalenEstimatorPolars)
    SurvivalStatisticsFactory.register("nelson_aalen_estimator", "spark", sp.NelsonAalenEstimatorSpark)
    SurvivalStatisticsFactory.register("kernel_hazard_smoother", "pandas", p.KernelHazardSmootherPandas)
    SurvivalStatisticsFactory.register("kernel_hazard_smoother", "polars", pl.KernelHazardSmootherPolars)
    SurvivalStatisticsFactory.register("kernel_hazard_smoother", "spark", sp.KernelHazardSmootherSpark)
    SurvivalStatisticsFactory.register("hazard_rate_calculator", "pandas", p.HazardRateCalculatorPandas)
    SurvivalStatisticsFactory.register("hazard_rate_calculator", "polars", pl.HazardRateCalculatorPolars)
    SurvivalStatisticsFactory.register("hazard_rate_calculator", "spark", sp.HazardRateCalculatorSpark)
    SurvivalStatisticsFactory.register("kaplan_meier_estimator", "pandas", p.KaplanMeierEstimatorPandas)
    SurvivalStatisticsFactory.register("kaplan_meier_estimator", "polars", pl.KaplanMeierEstimatorPolars)
    SurvivalStatisticsFactory.register("kaplan_meier_estimator", "spark", sp.KaplanMeierEstimatorSpark)
    SurvivalStatisticsFactory.register("median_survival_estimator", "pandas", p.MedianSurvivalEstimatorPandas)
    SurvivalStatisticsFactory.register("median_survival_estimator", "polars", pl.MedianSurvivalEstimatorPolars)
    SurvivalStatisticsFactory.register("median_survival_estimator", "spark", sp.MedianSurvivalEstimatorSpark)
    SurvivalStatisticsFactory.register("kaplan_meier_calculator", "pandas", p.KaplanMeierCalculatorPandas)
    SurvivalStatisticsFactory.register("kaplan_meier_calculator", "polars", pl.KaplanMeierCalculatorPolars)
    SurvivalStatisticsFactory.register("kaplan_meier_calculator", "spark", sp.KaplanMeierCalculatorSpark)
    SurvivalStatisticsFactory.register("time_to_event_summary_calculator", "pandas", p.TimeToEventSummaryCalculatorPandas)
    SurvivalStatisticsFactory.register("time_to_event_summary_calculator", "polars", pl.TimeToEventSummaryCalculatorPolars)
    SurvivalStatisticsFactory.register("time_to_event_summary_calculator", "spark", sp.TimeToEventSummaryCalculatorSpark)
    SurvivalStatisticsFactory.register("threshold_survival_analyser", "pandas", p.ThresholdSurvivalAnalyserPandas)
    SurvivalStatisticsFactory.register("threshold_survival_analyser", "polars", pl.ThresholdSurvivalAnalyserPolars)
    SurvivalStatisticsFactory.register("threshold_survival_analyser", "spark", sp.ThresholdSurvivalAnalyserSpark)
    SurvivalStatisticsFactory.register("exponential_fitter", "pandas", p.ExponentialFitterPandas)
    SurvivalStatisticsFactory.register("exponential_fitter", "polars", pl.ExponentialFitterPolars)
    SurvivalStatisticsFactory.register("exponential_fitter", "spark", sp.ExponentialFitterSpark)
    SurvivalStatisticsFactory.register("time_to_event_calculator", "pandas", p.TimeToEventCalculatorPandas)
    SurvivalStatisticsFactory.register("time_to_event_calculator", "polars", pl.TimeToEventCalculatorPolars)
    SurvivalStatisticsFactory.register("time_to_event_calculator", "spark", sp.TimeToEventCalculatorSpark)

_register()
