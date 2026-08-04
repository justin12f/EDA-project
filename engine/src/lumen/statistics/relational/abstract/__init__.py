"""Abstract contracts for the relational statistics domain."""
from lumen.statistics.relational.abstract.contingency_analysis import AbstractContingencyAnalysisCalculator
from lumen.statistics.relational.abstract.correlation_matrix import AbstractCorrelationMatrixCalculator
from lumen.statistics.relational.abstract.cross_correlation import AbstractCrossCorrelationCalculator
from lumen.statistics.relational.abstract.granger_causality import AbstractGrangerCausalityCalculator
from lumen.statistics.relational.abstract.interaction_effects import AbstractInteractionEffectsCalculator
from lumen.statistics.relational.abstract.multicollinearity import AbstractMulticollinearityCalculator
from lumen.statistics.relational.abstract.mutual_information import AbstractMutualInformationCalculator
from lumen.statistics.relational.abstract.partial_correlation import AbstractPartialCorrelationCalculator

__all__ = [
    "AbstractContingencyAnalysisCalculator",
    "AbstractCorrelationMatrixCalculator",
    "AbstractCrossCorrelationCalculator",
    "AbstractGrangerCausalityCalculator",
    "AbstractInteractionEffectsCalculator",
    "AbstractMulticollinearityCalculator",
    "AbstractMutualInformationCalculator",
    "AbstractPartialCorrelationCalculator",
]
