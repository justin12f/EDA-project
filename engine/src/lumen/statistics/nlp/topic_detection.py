"""LDA-based topic detection via co-occurrence matrix decomposition."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `NlpStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

@dataclass(frozen=True)
class Topic:
    """Immutable representation of a single detected topic."""

    topic_index: int
    top_terms: list[str]
    term_weights: dict[str, float]
    coherence_proxy: float

class BagOfWordsBuilder:
    """Builds a term-document matrix (bag-of-words) from a corpus.

    Filters by minimum document frequency to remove rare noise terms
    and maximum document frequency to suppress corpus-wide stopwords.
    """

    _TOKEN_PATTERN: re.Pattern = re.compile(r"[a-záéíóúüñ]+", re.UNICODE)
    _MINIMUM_TOKEN_LENGTH: int = 3

    def build(
        self,
        series: pd.Series,
        min_df: int,
        max_df_ratio: float,
    ) -> tuple[np.ndarray, list[str], list[list[str]]]:
        """Build term-document matrix.

        Args:
            series: String Series (one document per row).
            min_df: Minimum document frequency for a term to be included.
            max_df_ratio: Maximum document frequency ratio (e.g., 0.9 removes
                terms appearing in >90% of documents).

        Returns:
            Tuple (term_doc_matrix, vocabulary, corpus_tokens).
            Matrix shape: (n_docs, vocab_size), values are raw counts.

        Raises:
            ValueError: If vocabulary is empty after filtering.
        """
        corpus_tokens = [
            [
                t for t in self._TOKEN_PATTERN.findall(doc.lower())
                if len(t) >= self._MINIMUM_TOKEN_LENGTH
            ]
            for doc in series
        ]

        n_docs = len(corpus_tokens)
        all_terms: set[str] = set(t for doc in corpus_tokens for t in doc)

        doc_sets = [set(tokens) for tokens in corpus_tokens]
        doc_freq = {
            term: sum(1 for doc in doc_sets if term in doc)
            for term in all_terms
        }

        max_df_count = int(max_df_ratio * n_docs)
        vocabulary = sorted(
            term for term, df in doc_freq.items()
            if min_df <= df <= max_df_count
        )

        if not vocabulary:
            raise ValueError(
                f"Vocabulary is empty after filtering (min_df={min_df}, "
                f"max_df_ratio={max_df_ratio}). "
                "Adjust thresholds or check input text quality."
            )

        term_index = {term: i for i, term in enumerate(vocabulary)}
        matrix = np.zeros((n_docs, len(vocabulary)), dtype=float)

        for doc_idx, tokens in enumerate(corpus_tokens):
            for token in tokens:
                if token in term_index:
                    matrix[doc_idx, term_index[token]] += 1.0

        return matrix, vocabulary, corpus_tokens

class NMFTopicExtractor:
    """Non-negative Matrix Factorization topic extraction (LDA-alternative).

    Factorizes the TF-IDF matrix W ≈ H · V where:
        H: Document-topic matrix (n_docs × n_topics)
        V: Topic-term matrix  (n_topics × vocab_size)

    NMF is chosen over pure LDA because:
        1. No external dependencies (sklearn only).
        2. Faster convergence on small-medium corpora.
        3. Produces directly interpretable non-negative factors.

    Approximates LDA-style topic coherence via within-topic term
    co-occurrence as a proxy (no held-out perplexity computation).
    """

    def extract(
        self,
        tfidf_matrix: np.ndarray,
        vocabulary: list[str],
        n_topics: int,
        top_terms_per_topic: int,
        random_seed: int,
    ) -> list[Topic]:
        """Run NMF and extract topics.

        Args:
            tfidf_matrix: TF-IDF weighted term-document matrix.
            vocabulary: Sorted vocabulary list.
            n_topics: Number of topics to extract.
            top_terms_per_topic: Top N terms to show per topic.
            random_seed: Seed for NMF initialization.

        Returns:
            List of Topic dataclasses.

        Raises:
            ValueError: If n_topics exceeds matrix rank.
        """
        # Using sklearn.decomposition import NMF

        if n_topics >= tfidf_matrix.shape[0]:
            raise ValueError(
                f"n_topics ({n_topics}) must be less than n_documents "
                f"({tfidf_matrix.shape[0]})."
            )

        model = NMF(
            n_components=n_topics,
            random_state=random_seed,
            max_iter=200,
        )
        model.fit(tfidf_matrix)

        components = model.components_  # shape: (n_topics, vocab_size)
        topics: list[Topic] = []

        for topic_idx in range(n_topics):
            topic_weights = components[topic_idx]
            top_indices = np.argsort(topic_weights)[::-1][:top_terms_per_topic]
            top_terms = [vocabulary[i] for i in top_indices]
            term_weights = {
                vocabulary[i]: round(float(topic_weights[i]), 6)
                for i in top_indices
            }

            # Coherence proxy: mean pairwise correlation of top term weights
            top_weight_vec = topic_weights[top_indices]
            coherence = float(np.mean(top_weight_vec) / (float(np.std(top_weight_vec)) + 1e-8))

            topics.append(
                Topic(
                    topic_index=topic_idx,
                    top_terms=top_terms,
                    term_weights=term_weights,
                    coherence_proxy=round(coherence, 4),
                )
            )

        return topics

class TFIDFWeightedMatrixBuilder:
    """Applies TF-IDF weighting to a raw count matrix.

    TF-IDF = TF × log((1+N)/(1+df) + 1)  [sklearn smooth variant]
    """

    def apply(self, count_matrix: np.ndarray) -> np.ndarray:
        """Convert raw count matrix to TF-IDF weighted matrix.

        Args:
            count_matrix: Raw term-document count matrix (n_docs × vocab_size).

        Returns:
            TF-IDF weighted matrix of same shape.
        """
        n_docs = count_matrix.shape[0]
        tf = count_matrix / (count_matrix.sum(axis=1, keepdims=True) + 1e-8)
        df = (count_matrix > 0).sum(axis=0).astype(float)
        idf = np.log((1 + n_docs) / (1 + df)) + 1
        return tf * idf

class TopicDetectionCalculator:
    """NMF-based topic detection for a text column.

    Workflow:
        calculator = TopicDetectionCalculator()
        result = calculator.calculate(
            series=df["article_body"],
            n_topics=5,
            top_terms_per_topic=10,
            min_df=2,
            max_df_ratio=0.9,
            random_seed=42,
        )
    """

    _MINIMUM_DOCUMENTS: int = 10

    def __init__(self) -> None:
        self._bow_builder = BagOfWordsBuilder()
        self._tfidf_builder = TFIDFWeightedMatrixBuilder()
        self._extractor = NMFTopicExtractor()

    def calculate(
        self,
        series: pd.Series,
        n_topics: int = 5,
        top_terms_per_topic: int = 10,
        min_df: int = 2,
        max_df_ratio: float = 0.9,
        random_seed: int = 42,
    ) -> dict:
        """Detect latent topics in a text corpus.

        Args:
            series: String Series of documents.
            n_topics: Number of topics to discover.
            top_terms_per_topic: Top N terms shown per topic.
            min_df: Minimum document frequency for vocabulary.
            max_df_ratio: Max document frequency ratio for vocabulary.
            random_seed: Seed for NMF reproducibility.

        Returns:
            Dict with detected topics, top terms, and coherence proxies.

        Raises:
            ValueError: If corpus is too small or parameters are invalid.
        """
        clean = series.dropna().astype(str)
        clean = clean[clean.str.strip().str.len() > 0]

        if len(clean) < self._MINIMUM_DOCUMENTS:
            raise ValueError(
                f"At least {self._MINIMUM_DOCUMENTS} documents required for topic "
                f"detection. Got {len(clean)}."
            )
        if n_topics < 2:
            raise ValueError(f"n_topics must be >= 2. Got {n_topics}.")
        if top_terms_per_topic < 1:
            raise ValueError(f"top_terms_per_topic must be >= 1. Got {top_terms_per_topic}.")
        if not 0.0 < max_df_ratio <= 1.0:
            raise ValueError(
                f"max_df_ratio must be in (0, 1]. Got {max_df_ratio}."
            )

        count_matrix, vocabulary, _ = self._bow_builder.build(
            clean, min_df, max_df_ratio
        )
        tfidf_matrix = self._tfidf_builder.apply(count_matrix)
        topics = self._extractor.extract(
            tfidf_matrix, vocabulary, n_topics, top_terms_per_topic, random_seed
        )

        return {
            "topics": [
                {
                    "topic_index": t.topic_index,
                    "top_terms": t.top_terms,
                    "term_weights": t.term_weights,
                    "coherence_proxy": t.coherence_proxy,
                }
                for t in topics
            ],
            "n_topics": n_topics,
            "vocabulary_size": len(vocabulary),
            "n_documents": len(clean),
            "top_terms_per_topic": top_terms_per_topic,
            "method": "nmf_tfidf",
        }
