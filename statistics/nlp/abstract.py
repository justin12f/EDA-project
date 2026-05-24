"""Abstract statistics contracts — domain `nlp`."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
T = TypeVar('T')

class AbstractTrigramExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLanguageProfileLibrary(ABC, Generic[T]):

    @abstractmethod
    def all_profiles(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLanguageScorer(ABC, Generic[T]):

    @abstractmethod
    def score_all(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLanguageDetectionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDocumentEntityExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractEntityDensityComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNamedEntityDensityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTokenSentimentScorer(ABC, Generic[T]):

    @abstractmethod
    def score(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractPolarityNormalizer(ABC, Generic[T]):

    @abstractmethod
    def normalize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSubjectivityEstimator(ABC, Generic[T]):

    @abstractmethod
    def estimate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSentimentLabelAssigner(ABC, Generic[T]):

    @abstractmethod
    def assign(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSentimentAnalysisCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTextNormalizer(ABC, Generic[T]):

    @abstractmethod
    def normalize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractWordTokenizer(ABC, Generic[T]):

    @abstractmethod
    def tokenize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSentenceTokenizer(ABC, Generic[T]):

    @abstractmethod
    def tokenize(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractLexicalDensityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractDocumentStatsComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTextBasicStatsCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTFIDFVectorizer(ABC, Generic[T]):

    @abstractmethod
    def fit_transform(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCosineSimilarityComputer(ABC, Generic[T]):

    @abstractmethod
    def compute(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractSimilarityLabelAssigner(ABC, Generic[T]):

    @abstractmethod
    def assign(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTextSimilarityCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractBagOfWordsBuilder(ABC, Generic[T]):

    @abstractmethod
    def build(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractNMFTopicExtractor(ABC, Generic[T]):

    @abstractmethod
    def extract(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTFIDFWeightedMatrixBuilder(ABC, Generic[T]):

    @abstractmethod
    def apply(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTopicDetectionCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractStopwordFilter(ABC, Generic[T]):

    @abstractmethod
    def filter(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractCorpusTokenizer(ABC, Generic[T]):

    @abstractmethod
    def tokenize_corpus(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTFCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractIDFCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractTFIDFAggregator(ABC, Generic[T]):

    @abstractmethod
    def aggregate(self, data: T, column: str, **kwargs: Any) -> Any: ...

class AbstractWordFrequencyCalculator(ABC, Generic[T]):

    @abstractmethod
    def calculate(self, data: T, column: str, **kwargs: Any) -> Any: ...
