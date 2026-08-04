"""Abstract contracts for the inferential statistics domain."""
from inferential.abstract.anova import AbstractANOVACalculator
from inferential.abstract.bootstrap import AbstractBootstrapEstimator
from inferential.abstract.chi_square import AbstractChiSquareCalculator
from inferential.abstract.confidence_intervals import AbstractConfidenceIntervalCalculator
from inferential.abstract.correlation_significance import AbstractCorrelationSignificanceCalculator
from inferential.abstract.effect_size import AbstractEffectSizeCalculator
from inferential.abstract.hypothesis_test import AbstractHypothesisTestSuite
from inferential.abstract.power_analysis import AbstractPowerAnalysisCalculator

__all__ = [
    "AbstractANOVACalculator",
    "AbstractBootstrapEstimator",
    "AbstractChiSquareCalculator",
    "AbstractConfidenceIntervalCalculator",
    "AbstractCorrelationSignificanceCalculator",
    "AbstractEffectSizeCalculator",
    "AbstractHypothesisTestSuite",
    "AbstractPowerAnalysisCalculator",
]
