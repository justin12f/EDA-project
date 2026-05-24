"""Abstract statistics contracts — domain `inferential`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractTukeyHSDPostHoc(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractOneWayAnovaCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBootstrapSampler(ABC, Generic[T]):

    @abstractmethod
    def generate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBootstrapStatisticEstimator(ABC, Generic[T]):

    @abstractmethod
    def estimate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPercentilesBootstrapCI(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBootstrapEstimator(ABC, Generic[T]):

    @abstractmethod
    def estimate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractContingencyTableBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCramersVCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractChiSquareTestCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseConfidenceInterval(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMeanConfidenceInterval(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractProportionConfidenceInterval(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMeanDifferenceConfidenceInterval(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractConfidenceIntervalCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCorrelationInterpreter(ABC, Generic[T]):

    @abstractmethod
    def interpret(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFisherZTransformer(ABC, Generic[T]):

    @abstractmethod
    def confidence_interval(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCorrelationSignificanceCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEffectSizeInterpreter(ABC, Generic[T]):

    @abstractmethod
    def interpret_cohens_d(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def interpret_cramers_v(self, data: T, column: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def interpret_eta_squared(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCohensDCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEtaSquaredCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEffectSizeCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHypothesisInterpreter(ABC, Generic[T]):

    @abstractmethod
    def interpret(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseHypothesisTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMannWhitneyTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractWilcoxonTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractHypothesisTestSuite(ABC, Generic[T]):

    @abstractmethod
    def run(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMinimumSampleSizeCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractObservedPowerCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPowerAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
