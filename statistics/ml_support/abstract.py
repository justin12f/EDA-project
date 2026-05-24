"""Abstract statistics contracts — domain `ml_support`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractGiniImpurityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractImbalanceRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractStrategyAdvisor(ABC, Generic[T]):

    @abstractmethod
    def advise(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractClassImbalanceCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCVStrategySelector(ABC, Generic[T]):

    @abstractmethod
    def select(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCVConfidenceIntervalCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCrossValidationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCovarianceMatrixComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEigenDecompositionCalculator(ABC, Generic[T]):

    @abstractmethod
    def decompose(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractOptimalComponentSelector(ABC, Generic[T]):

    @abstractmethod
    def select(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDimensionalityReductionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseImportanceExtractor(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGiniImportanceExtractor(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPermutationImportanceExtractor(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractImportanceRanker(ABC, Generic[T]):

    @abstractmethod
    def rank(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFeatureImportanceCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseFeatureScoringMethod(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def score(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractChi2Scorer(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def score(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractANOVAFScorer(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def score(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMIScorer(ABC, Generic[T]):

    @abstractmethod
    def name(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def score(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractScoreRanker(ABC, Generic[T]):

    @abstractmethod
    def rank(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFeatureSelectionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractVarianceComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFrequencyRatioComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNearZeroVarianceClassifier(ABC, Generic[T]):

    @abstractmethod
    def classify(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFeatureVarianceCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBiasVarianceDiagnostic(ABC, Generic[T]):

    @abstractmethod
    def classify(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTrainingSizeGenerator(ABC, Generic[T]):

    @abstractmethod
    def generate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLearningCurveCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractResidualNormalityChecker(ABC, Generic[T]):

    @abstractmethod
    def check(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHomoscedasticityChecker(ABC, Generic[T]):

    @abstractmethod
    def check(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAutocorrelationChecker(ABC, Generic[T]):

    @abstractmethod
    def check(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractModelResidualsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
