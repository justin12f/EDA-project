"""Word frequency analysis: TF and TF-IDF scoring."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TermScore:
    """Immutable frequency/relevance record for a single term."""

    term: str
    tf: float
    df: int
    idf: float
    tfidf: float
    rank: int


class StopwordFilter:
    """Filters tokens against a built-in English stopword set.

    Designed to be easily extended: subclass and override
    `_STOPWORDS` or inject a custom set via the constructor.
    """

    _DEFAULT_STOPWORDS: frozenset[str] = frozenset({
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "of",
        "and", "or", "but", "for", "with", "as", "by", "from", "be",
        "was", "are", "were", "has", "have", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "not", "no", "this", "that", "these", "those", "he", "she", "they",
        "we", "you", "his", "her", "their", "our", "your", "its", "my",
        "me", "him", "us", "them", "what", "which", "who", "when", "where",
        "how", "all", "more", "other", "than", "then", "into", "over",
        "after", "before", "up", "so", "if", "about", "also", "just",
        "been", "very", "such", "same", "too", "however",
    })

    def __init__(self, custom_stopwords: frozenset[str] | None = None) -> None:
        self._stopwords = custom_stopwords or self._DEFAULT_STOPWORDS

    def filter(self, tokens: list[str]) -> list[str]:
        """Remove stopword tokens.

        Args:
            tokens: List of lowercase word tokens.

        Returns:
            Filtered token list.
        """
        return [t for t in tokens if t not in self._stopwords]


class CorpusTokenizer:
    """Tokenizes a Series of documents into per-document token lists."""

    _TOKEN_PATTERN: re.Pattern = re.compile(r"[a-záéíóúüñ']+", re.UNICODE)
    _MINIMUM_TOKEN_LENGTH: int = 2

    def tokenize_corpus(self, series: pd.Series) -> list[list[str]]:
        """Tokenize all documents in the corpus.

        Args:
            series: String Series (already cleaned/normalized).

        Returns:
            List of token lists, one per document.
        """
        return [
            [
                t for t in self._TOKEN_PATTERN.findall(doc.lower())
                if len(t) >= self._MINIMUM_TOKEN_LENGTH
            ]
            for doc in series
        ]


class TFCalculator:
    """Computes normalized term frequency for a single document.

    TF(t, d) = count(t, d) / |d|
    """

    def calculate(self, tokens: list[str]) -> dict[str, float]:
        """Compute TF for all terms in a document.

        Args:
            tokens: Token list for one document.

        Returns:
            Dict mapping term → TF value.
        """
        n = len(tokens)
        if n == 0:
            return {}
        counts = Counter(tokens)
        return {term: count / n for term, count in counts.items()}


class IDFCalculator:
    """Computes smoothed IDF across the corpus.

    IDF(t) = log((1 + N) / (1 + df(t))) + 1   [sklearn smooth variant]

    Smoothing prevents division by zero for unseen terms and reduces
    the weight of extremely common terms without zeroing them out.
    """

    def calculate(
        self,
        corpus_tokens: list[list[str]],
        vocabulary: set[str],
    ) -> dict[str, float]:
        """Compute IDF for all vocabulary terms.

        Args:
            corpus_tokens: Per-document token lists.
            vocabulary: Set of all unique terms in the corpus.

        Returns:
            Dict mapping term → IDF value.
        """
        n_docs = len(corpus_tokens)
        doc_sets = [set(tokens) for tokens in corpus_tokens]

        return {
            term: math.log((1 + n_docs) / (1 + sum(1 for doc in doc_sets if term in doc))) + 1
            for term in vocabulary
        }


class TFIDFAggregator:
    """Aggregates TF-IDF scores across all documents for corpus-level ranking.

    Corpus TF-IDF(t) = mean(TF(t, d)) × IDF(t)  for all d where t appears.
    """

    def aggregate(
        self,
        corpus_tokens: list[list[str]],
        idf: dict[str, float],
    ) -> dict[str, tuple[float, int, float]]:
        """Compute aggregated corpus-level TF-IDF.

        Args:
            corpus_tokens: Per-document token lists.
            idf: IDF mapping.

        Returns:
            Dict mapping term → (mean_tf, doc_frequency, tfidf_score).
        """
        tf_calc = TFCalculator()
        term_tf_accumulator: dict[str, list[float]] = {}

        for tokens in corpus_tokens:
            tf_doc = tf_calc.calculate(tokens)
            for term, tf_val in tf_doc.items():
                if term not in term_tf_accumulator:
                    term_tf_accumulator[term] = []
                term_tf_accumulator[term].append(tf_val)

        return {
            term: (
                sum(tfs) / len(tfs),
                len(tfs),
                (sum(tfs) / len(tfs)) * idf.get(term, 1.0),
            )
            for term, tfs in term_tf_accumulator.items()
        }


class WordFrequencyCalculator:
    """Corpus-level TF and TF-IDF ranking with stopword filtering.

    Workflow:
        calculator = WordFrequencyCalculator()
        result = calculator.calculate(
            series=df["review_text"],
            top_n=30,
            remove_stopwords=True,
            custom_stopwords=None,   # optional frozenset
        )
    """

    _MINIMUM_DOCUMENTS: int = 2

    def __init__(self) -> None:
        self._tokenizer = CorpusTokenizer()
        self._idf_calc = IDFCalculator()
        self._aggregator = TFIDFAggregator()

    def calculate(
        self,
        series: pd.Series,
        top_n: int = 30,
        remove_stopwords: bool = True,
        custom_stopwords: frozenset[str] | None = None,
    ) -> dict:
        """Compute word frequencies and TF-IDF scores.

        Args:
            series: String Series (one document per row).
            top_n: Number of top terms to return.
            remove_stopwords: Whether to filter stopwords.
            custom_stopwords: Optional custom stopword set.

        Returns:
            Dict with ranked terms by TF-IDF, raw TF, and corpus stats.

        Raises:
            ValueError: If corpus is too small.
        """
        clean = series.dropna().astype(str)
        clean = clean[clean.str.strip().str.len() > 0]

        if len(clean) < self._MINIMUM_DOCUMENTS:
            raise ValueError(
                f"At least {self._MINIMUM_DOCUMENTS} documents required for "
                f"TF-IDF. Got {len(clean)}."
            )

        corpus_tokens = self._tokenizer.tokenize_corpus(clean)

        if remove_stopwords:
            stopword_filter = StopwordFilter(custom_stopwords)
            corpus_tokens = [
                stopword_filter.filter(tokens) for tokens in corpus_tokens
            ]

        vocabulary = set(t for doc in corpus_tokens for t in doc)

        if not vocabulary:
            raise ValueError(
                "Vocabulary is empty after tokenization and stopword filtering. "
                "Check input text quality."
            )

        idf = self._idf_calc.calculate(corpus_tokens, vocabulary)
        aggregated = self._aggregator.aggregate(corpus_tokens, idf)

        scored: list[TermScore] = [
            TermScore(
                term=term,
                tf=round(mean_tf, 6),
                df=df_count,
                idf=round(idf.get(term, 1.0), 6),
                tfidf=round(tfidf_score, 6),
                rank=0,
            )
            for term, (mean_tf, df_count, tfidf_score) in aggregated.items()
        ]

        scored.sort(key=lambda s: s.tfidf, reverse=True)
        top_scored = scored[:top_n]
        ranked = [
            TermScore(
                term=s.term, tf=s.tf, df=s.df,
                idf=s.idf, tfidf=s.tfidf, rank=i + 1,
            )
            for i, s in enumerate(top_scored)
        ]

        total_terms = sum(len(doc) for doc in corpus_tokens)

        return {
            "terms": [
                {
                    "rank": s.rank,
                    "term": s.term,
                    "tf": s.tf,
                    "document_frequency": s.df,
                    "idf": s.idf,
                    "tfidf": s.tfidf,
                }
                for s in ranked
            ],
            "corpus_summary": {
                "n_documents": len(clean),
                "vocabulary_size": len(vocabulary),
                "total_tokens": total_terms,
                "stopwords_removed": remove_stopwords,
            },
            "top_n": top_n,
        }
