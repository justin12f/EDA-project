"""Abstract statistics contracts — domain `descriptive`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractMeanCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMedianCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractModeCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTrimmedMeanCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCentralTendencyInterpreter(ABC, Generic[T]):

    @abstractmethod
    def interpret(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCentralTendencyCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractVarianceCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractStandardDeviationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractRangeCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractIQRCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractMADCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCoefficientOfVariationCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDispersionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBimodalityDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTransformationAdvisor(ABC, Generic[T]):

    @abstractmethod
    def advise(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDistributionFitter(ABC, Generic[T]):

    @abstractmethod
    def fit_all(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDistributionClassifier(ABC, Generic[T]):

    @abstractmethod
    def classify(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBinCountSelector(ABC, Generic[T]):

    @abstractmethod
    def select(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFrequencyTableBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractFrequencyDistributionBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBaseNormalityTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractShapiroWilkTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractAndersonDarlingTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractKolmogorovSmirnovTest(ABC, Generic[T]):

    @abstractmethod
    def test(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNormalityTestSuite(ABC, Generic[T]):

    @abstractmethod
    def run(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPercentileOutlierDetector(ABC, Generic[T]):

    @abstractmethod
    def detect(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPercentilesCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSkewnessInterpreter(ABC, Generic[T]):

    @abstractmethod
    def interpret(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractKurtosisInterpreter(ABC, Generic[T]):

    @abstractmethod
    def interpret(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSkewnessKurtosisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractValueCountsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
