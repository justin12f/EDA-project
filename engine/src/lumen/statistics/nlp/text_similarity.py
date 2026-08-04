"""Document similarity: TF-IDF cosine similarity between text columns."""

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
class SimilarityResult:
    """Immutable pairwise or column-to-column similarity result."""

    document_index: int
    similarity: float
    label: str

class TFIDFVectorizer:
    """Minimal TF-IDF vectorizer for pair-level similarity.

    Builds vocabulary from both input columns combined, then
    represents each document as a normalized TF-IDF vector.
    Dependency-free — uses only numpy.
    """

    _TOKEN_PATTERN: re.Pattern = re.compile(r"[a-záéíóúüñ]+", re.UNICODE)
    _MINIMUM_TOKEN_LENGTH: int = 2

    def fit_transform(
        self, corpus_a: list[str], corpus_b: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit vocabulary on both corpora and transform to TF-IDF vectors.

        Args:
            corpus_a: List of documents from column A.
            corpus_b: List of documents from column B.

        Returns:
            Tuple (matrix_a, matrix_b) — each row is a normalized TF-IDF vector.
        """
        all_docs = corpus_a + corpus_b
        tokenized = [self._tokenize(doc) for doc in all_docs]
        vocabulary = self._build_vocabulary(tokenized)
        term_index = {term: i for i, term in enumerate(vocabulary)}

        n_docs = len(all_docs)
        df = self._compute_df(tokenized, vocabulary)
        idf = np.log((1 + n_docs) / (1 + df)) + 1

        matrix_a = self._build_tfidf_matrix(
            [self._tokenize(d) for d in corpus_a], term_index, idf
        )
        matrix_b = self._build_tfidf_matrix(
            [self._tokenize(d) for d in corpus_b], term_index, idf
        )

        matrix_a = self._l2_normalize(matrix_a)
        matrix_b = self._l2_normalize(matrix_b)

        return matrix_a, matrix_b

    def _tokenize(self, text: str) -> list[str]:
        return [
            t for t in self._TOKEN_PATTERN.findall(text.lower())
            if len(t) >= self._MINIMUM_TOKEN_LENGTH
        ]

    def _build_vocabulary(self, tokenized: list[list[str]]) -> list[str]:
        vocab: set[str] = set()
        for tokens in tokenized:
            vocab.update(tokens)
        return sorted(vocab)

    def _compute_df(
        self, tokenized: list[list[str]], vocabulary: list[str]
    ) -> np.ndarray:
        doc_sets = [set(tokens) for tokens in tokenized]
        return np.array([
            sum(1 for doc in doc_sets if term in doc)
            for term in vocabulary
        ], dtype=float)

    def _build_tfidf_matrix(
        self,
        tokenized: list[list[str]],
        term_index: dict[str, int],
        idf: np.ndarray,
    ) -> np.ndarray:
        matrix = np.zeros((len(tokenized), len(term_index)), dtype=float)
        for doc_idx, tokens in enumerate(tokenized):
            if not tokens:
                continue
            for token in tokens:
                if token in term_index:
                    matrix[doc_idx, term_index[token]] += 1.0
            matrix[doc_idx] = (matrix[doc_idx] / len(tokens)) * idf
        return matrix

    def _l2_normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

class CosineSimilarityComputer:
    """Computes row-wise cosine similarity between two L2-normalized matrices.

    For L2-normalized vectors: cos(θ) = a · b = dot product.
    Result is clipped to [0, 1] to handle floating-point imprecision.
    """

    def compute(
        self, matrix_a: np.ndarray, matrix_b: np.ndarray
    ) -> np.ndarray:
        """Compute element-wise cosine similarity.

        Args:
            matrix_a: L2-normalized matrix (n × vocab).
            matrix_b: L2-normalized matrix (n × vocab).

        Returns:
            1D array of cosine similarity scores in [0, 1].
        """
        return np.clip(
            np.sum(matrix_a * matrix_b, axis=1), 0.0, 1.0
        )

class SimilarityLabelAssigner:
    """Assigns a qualitative label to a cosine similarity score.

    Thresholds (commonly used in information retrieval):
        >= 0.9: very_similar
        >= 0.7: similar
        >= 0.4: somewhat_similar
        < 0.4:  dissimilar
    """

    _THRESHOLDS: list[tuple[float, str]] = [
        (0.9, "very_similar"),
        (0.7, "similar"),
        (0.4, "somewhat_similar"),
        (0.0, "dissimilar"),
    ]

    def assign(self, similarity: float) -> str:
        """Assign label based on similarity score.

        Args:
            similarity: Cosine similarity in [0, 1].

        Returns:
            Label string.
        """
        for threshold, label in self._THRESHOLDS:
            if similarity >= threshold:
                return label
        return "dissimilar"

class TextSimilarityCalculator:
    """Pairwise TF-IDF cosine similarity between two text columns.

    Workflow:
        calculator = TextSimilarityCalculator()
        result = calculator.calculate(
            data_frame=df,
            column_a="original_text",
            column_b="translated_text",
        )
    """

    _MINIMUM_DOCUMENTS: int = 1

    def __init__(self) -> None:
        self._vectorizer = TFIDFVectorizer()
        self._cosine_computer = CosineSimilarityComputer()
        self._label_assigner = SimilarityLabelAssigner()

    def calculate(
        self,
        data_frame: pd.DataFrame,
        column_a: str,
        column_b: str,
    ) -> dict:
        """Compute pairwise text similarity between two columns.

        Args:
            data_frame: Source DataFrame.
            column_a: First text column name.
            column_b: Second text column name.

        Returns:
            Dict with per-row similarities, distribution summary, and statistics.

        Raises:
            KeyError: If columns are not found.
            ValueError: If no valid document pairs exist.
        """
        for col in (column_a, column_b):
            if col not in data_frame.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

        paired = data_frame[[column_a, column_b]].dropna()
        paired = paired[
            paired[column_a].astype(str).str.strip().str.len() > 0
            & paired[column_b].astype(str).str.strip().str.len() > 0
        ]

        if len(paired) < self._MINIMUM_DOCUMENTS:
            raise ValueError(
                "No valid document pairs found after dropping nulls and empty strings."
            )

        corpus_a = paired[column_a].astype(str).tolist()
        corpus_b = paired[column_b].astype(str).tolist()

        matrix_a, matrix_b = self._vectorizer.fit_transform(corpus_a, corpus_b)
        similarities = self._cosine_computer.compute(matrix_a, matrix_b)

        results: list[SimilarityResult] = [
            SimilarityResult(
                document_index=idx,
                similarity=round(float(sim), 6),
                label=self._label_assigner.assign(float(sim)),
            )
            for idx, sim in enumerate(similarities)
        ]

        label_counts: dict[str, int] = {}
        for r in results:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1

        sim_array = np.array([r.similarity for r in results])

        return {
            "similarities": [
                {
                    "document_index": r.document_index,
                    "similarity": r.similarity,
                    "label": r.label,
                }
                for r in results
            ],
            "summary": {
                "mean_similarity": round(float(sim_array.mean()), 4),
                "std_similarity": round(float(sim_array.std()), 4),
                "min_similarity": round(float(sim_array.min()), 4),
                "max_similarity": round(float(sim_array.max()), 4),
                "median_similarity": round(float(np.median(sim_array)), 4),
                "label_distribution": label_counts,
            },
            "column_a": column_a,
            "column_b": column_b,
            "n_pairs": len(results),
        }
