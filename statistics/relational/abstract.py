"""Abstract statistics contracts — domain `relational`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractOddsRatioCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRelativeRiskCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractContingencyAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCorrelationMatrixComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCorrelationPairExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHighCorrelationFlagDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCorrelationMatrixCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSeriesStandardizer(ABC, Generic[T]):

    @abstractmethod
    def standardize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLaggedCorrelationComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPeakLagDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCrossCorrelationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLaggedDesignMatrixBuilder(ABC, Generic[T]):

    @abstractmethod
    def build_restricted(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def build_unrestricted(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractOLSResidualCalculator(ABC, Generic[T]):

    @abstractmethod
    def residual_ss(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGrangerFStatisticCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractGrangerCausalityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAdditiveModelEvaluator(ABC, Generic[T]):

    @abstractmethod
    def evaluate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractInteractionModelEvaluator(ABC, Generic[T]):

    @abstractmethod
    def evaluate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractInteractionGainClassifier(ABC, Generic[T]):

    @abstractmethod
    def is_meaningful(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractInteractionEffectsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractVIFRiskClassifier(ABC, Generic[T]):

    @abstractmethod
    def classify(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSingleFeatureVIFCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMulticollinearityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTargetTypeDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMIScoreNormalizer(ABC, Generic[T]):

    @abstractmethod
    def normalize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMutualInformationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractResidualExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPartialCorrelationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
