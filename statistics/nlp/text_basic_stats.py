"""Basic text statistics: length, lexical density, vocabulary richness."""

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

@dataclass(frozen=True)
class DocumentStats:
    """Immutable statistics for a single text document."""

    document_index: int
    char_count: int
    word_count: int
    unique_word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float
    lexical_density: float
    type_token_ratio: float

class TextNormalizer:
    """Normalizes raw text for consistent tokenization.

    Converts to lowercase, collapses whitespace, and strips
    leading/trailing spaces. Does NOT remove punctuation —
    sentence splitting depends on it.
    """

    _WHITESPACE_PATTERN: re.Pattern = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        """Normalize text for tokenization.

        Args:
            text: Raw document string.

        Returns:
            Normalized lowercase string.
        """
        return self._WHITESPACE_PATTERN.sub(" ", text.lower()).strip()

class WordTokenizer:
    """Tokenizes normalized text into word tokens.

    Splits on non-alphabetic characters. Filters tokens shorter
    than a minimum length to remove punctuation fragments.
    """

    _TOKEN_PATTERN: re.Pattern = re.compile(r"[a-záéíóúüñ']+", re.UNICODE)
    _MINIMUM_TOKEN_LENGTH: int = 2

    def tokenize(self, text: str) -> list[str]:
        """Extract word tokens from normalized text.

        Args:
            text: Normalized document string.

        Returns:
            List of word token strings.
        """
        return [
            t for t in self._TOKEN_PATTERN.findall(text)
            if len(t) >= self._MINIMUM_TOKEN_LENGTH
        ]

class SentenceTokenizer:
    """Splits text into sentences using punctuation boundaries."""

    _SENTENCE_PATTERN: re.Pattern = re.compile(
        r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚ\"])", re.UNICODE
    )

    def tokenize(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Raw or normalized document string.

        Returns:
            List of sentence strings.
        """
        sentences = self._SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

class LexicalDensityCalculator:
    """Computes lexical density: content words / total words.

    Approximates content words as tokens not in a basic stopword list.
    Higher density = more informative, denser text.
    Lower density = more functional/conversational text.
    """

    _ENGLISH_STOPWORDS: frozenset[str] = frozenset({
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "of",
        "and", "or", "but", "for", "with", "as", "by", "from", "be",
        "was", "are", "were", "been", "has", "have", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "this", "that", "these", "those", "he", "she", "they", "we", "you",
        "his", "her", "their", "our", "your", "its", "my", "me", "him",
        "us", "them", "what", "which", "who", "whom", "when", "where",
        "why", "how", "all", "each", "every", "more", "most", "other",
        "than", "then", "into", "over", "after", "before", "up",
    })

    def calculate(self, tokens: list[str]) -> float:
        """Compute lexical density.

        Args:
            tokens: Word token list.

        Returns:
            Lexical density in [0, 1]. 0 if tokens is empty.
        """
        if not tokens:
            return 0.0
        content_words = [t for t in tokens if t not in self._ENGLISH_STOPWORDS]
        return len(content_words) / len(tokens)

class DocumentStatsComputer:
    """Computes all statistics for a single document string."""

    def __init__(self) -> None:
        self._normalizer = TextNormalizer()
        self._word_tokenizer = WordTokenizer()
        self._sentence_tokenizer = SentenceTokenizer()
        self._lexical_density_calc = LexicalDensityCalculator()

    def compute(self, text: str, index: int) -> DocumentStats:
        """Compute all stats for a document.

        Args:
            text: Raw document string.
            index: Document index for identification.

        Returns:
            DocumentStats dataclass with all metrics.
        """
        normalized = self._normalizer.normalize(text)
        words = self._word_tokenizer.tokenize(normalized)
        sentences = self._sentence_tokenizer.tokenize(text)

        word_count = len(words)
        unique_words = set(words)
        sentence_count = max(len(sentences), 1)

        avg_word_length = (
            float(np.mean([len(w) for w in words])) if words else 0.0
        )
        avg_sentence_length = word_count / sentence_count
        lexical_density = self._lexical_density_calc.calculate(words)
        ttr = len(unique_words) / word_count if word_count > 0 else 0.0

        return DocumentStats(
            document_index=index,
            char_count=len(text),
            word_count=word_count,
            unique_word_count=len(unique_words),
            sentence_count=sentence_count,
            avg_word_length=round(avg_word_length, 4),
            avg_sentence_length=round(avg_sentence_length, 4),
            lexical_density=round(lexical_density, 4),
            type_token_ratio=round(ttr, 4),
        )

class TextBasicStatsCalculator:
    """Computes per-document and corpus-level text statistics.

    Workflow:
        calculator = TextBasicStatsCalculator()
        result = calculator.calculate(
            series=df["review_text"],
            sample_n=None,   # optional, limits documents processed
        )
    """

    _MINIMUM_DOCUMENTS: int = 1

    def __init__(self) -> None:
        self._computer = DocumentStatsComputer()

    def calculate(
        self,
        series: pd.Series,
        sample_n: int | None = None,
    ) -> dict:
        """Compute text statistics across all documents.

        Args:
            series: String Series where each element is a document.
            sample_n: If set, randomly samples n documents.

        Returns:
            Dict with per-document stats and corpus-level aggregates.

        Raises:
            ValueError: If series has no valid text documents.
        """
        clean = series.dropna().astype(str)
        clean = clean[clean.str.strip().str.len() > 0]

        if len(clean) < self._MINIMUM_DOCUMENTS:
            raise ValueError(
                "At least 1 non-empty document is required. "
                "Series contains no valid text."
            )

        if sample_n is not None and sample_n < len(clean):
            clean = clean.sample(n=sample_n, random_state=42)

        doc_stats: list[DocumentStats] = [
            self._computer.compute(text, idx)
            for idx, text in enumerate(clean)
        ]

        word_counts = np.array([d.word_count for d in doc_stats])
        char_counts = np.array([d.char_count for d in doc_stats])
        ttr_values = np.array([d.type_token_ratio for d in doc_stats])
        lex_density = np.array([d.lexical_density for d in doc_stats])

        return {
            "documents": [
                {
                    "document_index": d.document_index,
                    "char_count": d.char_count,
                    "word_count": d.word_count,
                    "unique_word_count": d.unique_word_count,
                    "sentence_count": d.sentence_count,
                    "avg_word_length": d.avg_word_length,
                    "avg_sentence_length": d.avg_sentence_length,
                    "lexical_density": d.lexical_density,
                    "type_token_ratio": d.type_token_ratio,
                }
                for d in doc_stats
            ],
            "corpus_summary": {
                "n_documents": len(doc_stats),
                "total_words": int(word_counts.sum()),
                "total_chars": int(char_counts.sum()),
                "mean_words_per_doc": round(float(word_counts.mean()), 2),
                "std_words_per_doc": round(float(word_counts.std()), 2),
                "median_words_per_doc": round(float(np.median(word_counts)), 2),
                "mean_type_token_ratio": round(float(ttr_values.mean()), 4),
                "mean_lexical_density": round(float(lex_density.mean()), 4),
            },
        }
