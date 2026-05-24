"""Polars statistics backends — `nlp`."""
from __future__ import annotations
from typing import Any
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.core.polars_frame import eager, numeric_series
from statistics.nlp.backends import pandas_impl
from statistics.nlp.backends.pandas_impl import *

from statistics.nlp.abstract import *

class TrigramExtractorPolars(AbstractTrigramExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TrigramExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class LanguageProfileLibraryPolars(AbstractLanguageProfileLibrary[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LanguageProfileLibraryPandas()

    def all_profiles(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.all_profiles(data, column, **kwargs)

class LanguageScorerPolars(AbstractLanguageScorer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LanguageScorerPandas()

    def score_all(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score_all(data, column, **kwargs)

class LanguageDetectionCalculatorPolars(AbstractLanguageDetectionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LanguageDetectionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DocumentEntityExtractorPolars(AbstractDocumentEntityExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DocumentEntityExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class EntityDensityComputerPolars(AbstractEntityDensityComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = EntityDensityComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class NamedEntityDensityCalculatorPolars(AbstractNamedEntityDensityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NamedEntityDensityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TokenSentimentScorerPolars(AbstractTokenSentimentScorer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TokenSentimentScorerPandas()

    def score(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class PolarityNormalizerPolars(AbstractPolarityNormalizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = PolarityNormalizerPandas()

    def normalize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class SubjectivityEstimatorPolars(AbstractSubjectivityEstimator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SubjectivityEstimatorPandas()

    def estimate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class SentimentLabelAssignerPolars(AbstractSentimentLabelAssigner[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SentimentLabelAssignerPandas()

    def assign(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class SentimentAnalysisCalculatorPolars(AbstractSentimentAnalysisCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SentimentAnalysisCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TextNormalizerPolars(AbstractTextNormalizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TextNormalizerPandas()

    def normalize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class WordTokenizerPolars(AbstractWordTokenizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = WordTokenizerPandas()

    def tokenize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.tokenize(data, column, **kwargs)

class SentenceTokenizerPolars(AbstractSentenceTokenizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SentenceTokenizerPandas()

    def tokenize(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.tokenize(data, column, **kwargs)

class LexicalDensityCalculatorPolars(AbstractLexicalDensityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = LexicalDensityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DocumentStatsComputerPolars(AbstractDocumentStatsComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = DocumentStatsComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class TextBasicStatsCalculatorPolars(AbstractTextBasicStatsCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TextBasicStatsCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TFIDFVectorizerPolars(AbstractTFIDFVectorizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TFIDFVectorizerPandas()

    def fit_transform(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.fit_transform(data, column, **kwargs)

class CosineSimilarityComputerPolars(AbstractCosineSimilarityComputer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CosineSimilarityComputerPandas()

    def compute(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class SimilarityLabelAssignerPolars(AbstractSimilarityLabelAssigner[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = SimilarityLabelAssignerPandas()

    def assign(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class TextSimilarityCalculatorPolars(AbstractTextSimilarityCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TextSimilarityCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BagOfWordsBuilderPolars(AbstractBagOfWordsBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = BagOfWordsBuilderPandas()

    def build(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class NMFTopicExtractorPolars(AbstractNMFTopicExtractor[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = NMFTopicExtractorPandas()

    def extract(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class TFIDFWeightedMatrixBuilderPolars(AbstractTFIDFWeightedMatrixBuilder[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TFIDFWeightedMatrixBuilderPandas()

    def apply(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.apply(data, column, **kwargs)

class TopicDetectionCalculatorPolars(AbstractTopicDetectionCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TopicDetectionCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class StopwordFilterPolars(AbstractStopwordFilter[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = StopwordFilterPandas()

    def filter(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.filter(data, column, **kwargs)

class CorpusTokenizerPolars(AbstractCorpusTokenizer[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = CorpusTokenizerPandas()

    def tokenize_corpus(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.tokenize_corpus(data, column, **kwargs)

class TFCalculatorPolars(AbstractTFCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TFCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class IDFCalculatorPolars(AbstractIDFCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = IDFCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TFIDFAggregatorPolars(AbstractTFIDFAggregator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = TFIDFAggregatorPandas()

    def aggregate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.aggregate(data, column, **kwargs)

class WordFrequencyCalculatorPolars(AbstractWordFrequencyCalculator[pl.DataFrame]):
    def __init__(self) -> None:
        self._pandas = WordFrequencyCalculatorPandas()

    def calculate(self, data: pl.DataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
