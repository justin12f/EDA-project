"""Spark statistics backends — `nlp`."""
from __future__ import annotations
from typing import Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from statistics.nlp.abstract import *

from statistics.nlp.backends import pandas_impl
from statistics.nlp.backends.pandas_impl import *

class TrigramExtractorSpark(AbstractTrigramExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TrigramExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class LanguageProfileLibrarySpark(AbstractLanguageProfileLibrary[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LanguageProfileLibraryPandas()

    def all_profiles(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.all_profiles(data, column, **kwargs)

class LanguageScorerSpark(AbstractLanguageScorer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LanguageScorerPandas()

    def score_all(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score_all(data, column, **kwargs)

class LanguageDetectionCalculatorSpark(AbstractLanguageDetectionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LanguageDetectionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DocumentEntityExtractorSpark(AbstractDocumentEntityExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DocumentEntityExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class EntityDensityComputerSpark(AbstractEntityDensityComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = EntityDensityComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class NamedEntityDensityCalculatorSpark(AbstractNamedEntityDensityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NamedEntityDensityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TokenSentimentScorerSpark(AbstractTokenSentimentScorer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TokenSentimentScorerPandas()

    def score(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.score(data, column, **kwargs)

class PolarityNormalizerSpark(AbstractPolarityNormalizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = PolarityNormalizerPandas()

    def normalize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class SubjectivityEstimatorSpark(AbstractSubjectivityEstimator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SubjectivityEstimatorPandas()

    def estimate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.estimate(data, column, **kwargs)

class SentimentLabelAssignerSpark(AbstractSentimentLabelAssigner[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SentimentLabelAssignerPandas()

    def assign(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class SentimentAnalysisCalculatorSpark(AbstractSentimentAnalysisCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SentimentAnalysisCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TextNormalizerSpark(AbstractTextNormalizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TextNormalizerPandas()

    def normalize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.normalize(data, column, **kwargs)

class WordTokenizerSpark(AbstractWordTokenizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = WordTokenizerPandas()

    def tokenize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.tokenize(data, column, **kwargs)

class SentenceTokenizerSpark(AbstractSentenceTokenizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SentenceTokenizerPandas()

    def tokenize(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.tokenize(data, column, **kwargs)

class LexicalDensityCalculatorSpark(AbstractLexicalDensityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = LexicalDensityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class DocumentStatsComputerSpark(AbstractDocumentStatsComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = DocumentStatsComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class TextBasicStatsCalculatorSpark(AbstractTextBasicStatsCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TextBasicStatsCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TFIDFVectorizerSpark(AbstractTFIDFVectorizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TFIDFVectorizerPandas()

    def fit_transform(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.fit_transform(data, column, **kwargs)

class CosineSimilarityComputerSpark(AbstractCosineSimilarityComputer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CosineSimilarityComputerPandas()

    def compute(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.compute(data, column, **kwargs)

class SimilarityLabelAssignerSpark(AbstractSimilarityLabelAssigner[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = SimilarityLabelAssignerPandas()

    def assign(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.assign(data, column, **kwargs)

class TextSimilarityCalculatorSpark(AbstractTextSimilarityCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TextSimilarityCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class BagOfWordsBuilderSpark(AbstractBagOfWordsBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = BagOfWordsBuilderPandas()

    def build(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.build(data, column, **kwargs)

class NMFTopicExtractorSpark(AbstractNMFTopicExtractor[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = NMFTopicExtractorPandas()

    def extract(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.extract(data, column, **kwargs)

class TFIDFWeightedMatrixBuilderSpark(AbstractTFIDFWeightedMatrixBuilder[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TFIDFWeightedMatrixBuilderPandas()

    def apply(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.apply(data, column, **kwargs)

class TopicDetectionCalculatorSpark(AbstractTopicDetectionCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TopicDetectionCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class StopwordFilterSpark(AbstractStopwordFilter[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = StopwordFilterPandas()

    def filter(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.filter(data, column, **kwargs)

class CorpusTokenizerSpark(AbstractCorpusTokenizer[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = CorpusTokenizerPandas()

    def tokenize_corpus(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.tokenize_corpus(data, column, **kwargs)

class TFCalculatorSpark(AbstractTFCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TFCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class IDFCalculatorSpark(AbstractIDFCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = IDFCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)

class TFIDFAggregatorSpark(AbstractTFIDFAggregator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = TFIDFAggregatorPandas()

    def aggregate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.aggregate(data, column, **kwargs)

class WordFrequencyCalculatorSpark(AbstractWordFrequencyCalculator[SparkDataFrame]):
    def __init__(self) -> None:
        self._pandas = WordFrequencyCalculatorPandas()

    def calculate(self, data: SparkDataFrame, column: str, **kwargs: Any) -> Any:
        return self._pandas.calculate(data, column, **kwargs)
