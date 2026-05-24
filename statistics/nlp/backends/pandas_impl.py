"""Pandas statistics backends — `nlp`."""
from __future__ import annotations
from typing import Any
import pandas as pd
from statistics.core.frame_extract import column_to_numpy
from statistics.nlp.abstract import *

import statistics.nlp.language_detection as _mod_language_detection
import statistics.nlp.named_entity_density as _mod_named_entity_density
import statistics.nlp.sentiment_analysis as _mod_sentiment_analysis
import statistics.nlp.text_basic_stats as _mod_text_basic_stats
import statistics.nlp.text_similarity as _mod_text_similarity
import statistics.nlp.topic_detection as _mod_topic_detection
import statistics.nlp.word_frequency as _mod_word_frequency

class TrigramExtractorPandas(AbstractTrigramExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_language_detection.TrigramExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class LanguageProfileLibraryPandas(AbstractLanguageProfileLibrary[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_language_detection.LanguageProfileLibrary()

    def all_profiles(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.all_profiles(arr, **kwargs)

class LanguageScorerPandas(AbstractLanguageScorer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_language_detection.LanguageScorer()

    def score_all(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.score_all(arr, **kwargs)

class LanguageDetectionCalculatorPandas(AbstractLanguageDetectionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_language_detection.LanguageDetectionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class DocumentEntityExtractorPandas(AbstractDocumentEntityExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_named_entity_density.DocumentEntityExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class EntityDensityComputerPandas(AbstractEntityDensityComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_named_entity_density.EntityDensityComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class NamedEntityDensityCalculatorPandas(AbstractNamedEntityDensityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_named_entity_density.NamedEntityDensityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TokenSentimentScorerPandas(AbstractTokenSentimentScorer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_sentiment_analysis.TokenSentimentScorer()

    def score(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.score(arr, **kwargs)

class PolarityNormalizerPandas(AbstractPolarityNormalizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_sentiment_analysis.PolarityNormalizer()

    def normalize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.normalize(arr, **kwargs)

class SubjectivityEstimatorPandas(AbstractSubjectivityEstimator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_sentiment_analysis.SubjectivityEstimator()

    def estimate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.estimate(arr, **kwargs)

class SentimentLabelAssignerPandas(AbstractSentimentLabelAssigner[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_sentiment_analysis.SentimentLabelAssigner()

    def assign(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.assign(arr, **kwargs)

class SentimentAnalysisCalculatorPandas(AbstractSentimentAnalysisCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_sentiment_analysis.SentimentAnalysisCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TextNormalizerPandas(AbstractTextNormalizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_basic_stats.TextNormalizer()

    def normalize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.normalize(arr, **kwargs)

class WordTokenizerPandas(AbstractWordTokenizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_basic_stats.WordTokenizer()

    def tokenize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.tokenize(arr, **kwargs)

class SentenceTokenizerPandas(AbstractSentenceTokenizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_basic_stats.SentenceTokenizer()

    def tokenize(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.tokenize(arr, **kwargs)

class LexicalDensityCalculatorPandas(AbstractLexicalDensityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_basic_stats.LexicalDensityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class DocumentStatsComputerPandas(AbstractDocumentStatsComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_basic_stats.DocumentStatsComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class TextBasicStatsCalculatorPandas(AbstractTextBasicStatsCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_basic_stats.TextBasicStatsCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TFIDFVectorizerPandas(AbstractTFIDFVectorizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_similarity.TFIDFVectorizer()

    def fit_transform(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.fit_transform(arr, **kwargs)

class CosineSimilarityComputerPandas(AbstractCosineSimilarityComputer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_similarity.CosineSimilarityComputer()

    def compute(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.compute(arr, **kwargs)

class SimilarityLabelAssignerPandas(AbstractSimilarityLabelAssigner[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_similarity.SimilarityLabelAssigner()

    def assign(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.assign(arr, **kwargs)

class TextSimilarityCalculatorPandas(AbstractTextSimilarityCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_text_similarity.TextSimilarityCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class BagOfWordsBuilderPandas(AbstractBagOfWordsBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_topic_detection.BagOfWordsBuilder()

    def build(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.build(arr, **kwargs)

class NMFTopicExtractorPandas(AbstractNMFTopicExtractor[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_topic_detection.NMFTopicExtractor()

    def extract(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.extract(arr, **kwargs)

class TFIDFWeightedMatrixBuilderPandas(AbstractTFIDFWeightedMatrixBuilder[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_topic_detection.TFIDFWeightedMatrixBuilder()

    def apply(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.apply(arr, **kwargs)

class TopicDetectionCalculatorPandas(AbstractTopicDetectionCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_topic_detection.TopicDetectionCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class StopwordFilterPandas(AbstractStopwordFilter[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_word_frequency.StopwordFilter()

    def filter(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.filter(arr, **kwargs)

class CorpusTokenizerPandas(AbstractCorpusTokenizer[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_word_frequency.CorpusTokenizer()

    def tokenize_corpus(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.tokenize_corpus(arr, **kwargs)

class TFCalculatorPandas(AbstractTFCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_word_frequency.TFCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class IDFCalculatorPandas(AbstractIDFCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_word_frequency.IDFCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)

class TFIDFAggregatorPandas(AbstractTFIDFAggregator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_word_frequency.TFIDFAggregator()

    def aggregate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.aggregate(arr, **kwargs)

class WordFrequencyCalculatorPandas(AbstractWordFrequencyCalculator[pd.DataFrame]):
    def __init__(self) -> None:
        self._legacy = _mod_word_frequency.WordFrequencyCalculator()

    def calculate(self, data: pd.DataFrame, column: str, **kwargs: Any) -> Any:
        arr = column_to_numpy(data, column)
        return self._legacy.calculate(arr, **kwargs)
