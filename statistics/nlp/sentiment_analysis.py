"""Lexicon-based sentiment analysis with VADER-style scoring."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `NlpStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations
import math
import re
from dataclasses import dataclass

import pandas as pd

@dataclass(frozen=True)
class SentimentResult:
    """Immutable sentiment result for a single document."""

    document_index: int
    polarity_score: float
    subjectivity_score: float
    label: str
    positive_word_count: int
    negative_word_count: int
    word_count: int

class SentimentLexicon:
    """Minimal built-in sentiment lexicon for dependency-free analysis.

    Covers ~200 high-signal positive and negative English terms.
    Extend by subclassing and overriding `_POSITIVE_TERMS`
    and `_NEGATIVE_TERMS`.
    """
    POSITIVE_TERMS: frozenset[str] = frozenset({
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "outstanding", "brilliant", "superb", "perfect", "best", "love",
        "loved", "enjoy", "enjoyed", "beautiful", "awesome", "impressive",
        "exceptional", "delightful", "pleasant", "satisfying", "satisfied",
        "happy", "pleased", "glad", "thrilled", "excited", "positive",
        "helpful", "useful", "reliable", "efficient", "effective", "strong",
        "clean", "clear", "easy", "fast", "quick", "smooth", "comfortable",
        "recommended", "recommend", "worth", "valuable", "innovative",
        "creative", "professional", "quality", "premium", "superior",
        "affordable", "fair", "honest", "transparent", "trustworthy",
        "safe", "secure", "accurate", "correct", "right", "true",
        "success", "successful", "achieve", "accomplished", "improved",
        "improvement", "progress", "advance", "benefit", "gain", "win",
        "winning", "reward", "rewarding", "joy", "joyful", "bright",
        "lively", "energetic", "powerful", "smart", "intelligent", "clever",
        "insightful", "thoughtful", "caring", "kind", "generous", "warm",
    })

    NEGATIVE_TERMS: frozenset[str] = frozenset({
        "bad", "terrible", "horrible", "awful", "poor", "worst", "hate",
        "hated", "disappointing", "disappointed", "disappoints", "failure",
        "fail", "failed", "wrong", "incorrect",
        "useless", "unreliable", "slow", "difficult", "hard", "confusing",
        "confused", "frustrating", "frustrated", "annoying", "annoyed",
        "unhappy", "unhelpful", "negative", "problem", "issue", "error",
        "bug", "crash", "loss", "lose", "losing", "waste", "wasted",
        "expensive", "overpriced", "misleading", "dishonest", "unsafe",
        "dangerous", "risky", "harmful", "weak", "dirty", "ugly", "boring",
        "dull", "mediocre", "inferior", "inadequate", "insufficient",
        "incomplete", "inaccurate", "false", "fake", "scam", "fraud",
        "corrupt", "inefficient", "ineffective", "complicated", "complex",
        "broken", "damaged", "defective", "flawed", "rejected", "denied",
        "banned", "blocked", "missing", "absent", "lacking", "limited",
        "restricted", "unacceptable", "unbearable", "intolerable", 
        "disgusting", "irritating", "offensive", "rude", "disrespectful",
        "negligent", "careless", "incompetent", "unstable",
    })

    NEGATION_WORDS: frozenset[str] = frozenset({
        "not", "no", "never", "neither", "nor", "nothing", "nobody",
        "nowhere", "hardly", "barely", "scarcely", "without", "lack",
    })

    INTENSIFIER_MULTIPLIERS: dict[str, float] = {
        "very": 1.5,
        "extremely": 2.0,
        "really": 1.3,
        "quite": 1.2,
        "absolutely": 1.8,
        "incredibly": 1.7,
        "utterly": 1.6,
        "totally": 1.4,
        "highly": 1.4,
        "fairly": 0.8,
        "somewhat": 0.7,
        "slightly": 0.5,
        "barely": 0.3,
    }

class TokenSentimentScorer:
    """Scores individual tokens with negation and intensifier awareness.

    Scoring logic:
        1. Each positive token contributes +1.0.
        2. Each negative token contributes -1.0.
        3. If a negation word appears in the 3-token window before the
           sentiment word, the score is flipped.
        4. If an intensifier appears in the 2-token window before the
           sentiment word, the score is multiplied by its weight.
    """

    _NEGATION_WINDOW: int = 3
    _INTENSIFIER_WINDOW: int = 2

    def __init__(self, lexicon: SentimentLexicon) -> None:
        self._lexicon = lexicon

    def score(self, tokens: list[str]) -> tuple[float, int, int]:
        """Compute raw sentiment score for a token list.

        Args:
            tokens: Lowercase word token list.

        Returns:
            Tuple (raw_score, positive_count, negative_count).
        """
        raw_score = 0.0
        positive_count = 0
        negative_count = 0

        for i, token in enumerate(tokens):
            base_score: float | None = None

            if token in self._lexicon.POSITIVE_TERMS:
                base_score = 1.0
                positive_count += 1
            elif token in self._lexicon.NEGATIVE_TERMS:
                base_score = -1.0
                negative_count += 1

            if base_score is None:
                continue

            window_start = max(0, i - self._NEGATION_WINDOW)
            preceding = tokens[window_start:i]

            if any(w in self._lexicon.NEGATION_WORDS for w in preceding):
                base_score *= -1.0

            intensifier_window = tokens[max(0, i - self._INTENSIFIER_WINDOW):i]
            for word in reversed(intensifier_window):
                multiplier = self._lexicon.INTENSIFIER_MULTIPLIERS.get(word)
                if multiplier is not None:
                    base_score *= multiplier
                    break

            raw_score += base_score

        return raw_score, positive_count, negative_count

class PolarityNormalizer:
    """Normalizes raw sentiment score to [-1, 1] using tanh compression.

    tanh provides smooth saturation: large raw scores don't produce
    polarity values far outside [-1, 1], and the function is symmetric.
    """

    def normalize(self, raw_score: float, word_count: int) -> float:
        """Normalize raw score to [-1, 1].

        Args:
            raw_score: Sum of token sentiment scores.
            word_count: Total token count for scale normalization.

        Returns:
            Polarity in [-1, 1].
        """

        if word_count == 0:
            return 0.0
        normalized = raw_score / max(word_count, 1)
        return round(math.tanh(normalized), 6)

class SubjectivityEstimator:
    """Estimates subjectivity as the proportion of sentiment-bearing tokens.

    Subjectivity = (positive_count + negative_count) / word_count
    Higher subjectivity = more opinion-laden text.
    Lower subjectivity = more factual/neutral text.
    """

    def estimate(
        self, positive_count: int, negative_count: int, word_count: int
    ) -> float:
        """Estimate subjectivity score.

        Args:
            positive_count: Count of positive tokens.
            negative_count: Count of negative tokens.
            word_count: Total token count.

        Returns:
            Subjectivity in [0, 1].
        """
        if word_count == 0:
            return 0.0
        return round((positive_count + negative_count) / word_count, 6)

class SentimentLabelAssigner:
    """Assigns a sentiment label based on polarity and subjectivity.

    Labels:
        - 'positive':      polarity > threshold
        - 'negative':      polarity < -threshold
        - 'neutral':       |polarity| <= threshold
        - 'mixed':         high subjectivity but low polarity
    """

    _POLARITY_THRESHOLD: float = 0.05
    _HIGH_SUBJECTIVITY_THRESHOLD: float = 0.15

    def assign(self, polarity: float, subjectivity: float) -> str:
        """Assign sentiment label.

        Args:
            polarity: Normalized polarity in [-1, 1].
            subjectivity: Subjectivity in [0, 1].

        Returns:
            Sentiment label string.
        """
        if abs(polarity) <= self._POLARITY_THRESHOLD:
            if subjectivity > self._HIGH_SUBJECTIVITY_THRESHOLD:
                return "mixed"
            return "neutral"
        return "positive" if polarity > 0 else "negative"

class SentimentAnalysisCalculator:
    """Lexicon-based sentiment analysis for a text column.

    Dependency-free: uses a built-in lexicon with negation
    and intensifier handling. For production, replace
    SentimentLexicon with a larger VADER/AFINN lexicon.

    Workflow:
        calculator = SentimentAnalysisCalculator()
        result = calculator.calculate(
            series=df["review"],
            sample_n=None,   # optional
        )
    """

    _MINIMUM_DOCUMENTS: int = 1
    _TOKEN_PATTERN: re.Pattern = re.compile(r"[a-záéíóúüñ']+", re.UNICODE)

    def __init__(self) -> None:
        self._lexicon = SentimentLexicon()
        self._token_scorer = TokenSentimentScorer(self._lexicon)
        self._polarity_normalizer = PolarityNormalizer()
        self._subjectivity_estimator = SubjectivityEstimator()
        self._label_assigner = SentimentLabelAssigner()

    def calculate(
        self,
        series: pd.Series,
        sample_n: int | None = None,
    ) -> dict:
        """Run sentiment analysis on all documents.

        Args:
            series: String Series of documents.
            sample_n: Optional sample size limit.

        Returns:
            Dict with per-document results and corpus-level distribution.

        Raises:
            ValueError: If series contains no valid documents.
        """
        clean = series.dropna().astype(str)
        clean = clean[clean.str.strip().str.len() > 0]

        if len(clean) < self._MINIMUM_DOCUMENTS:
            raise ValueError(
                "At least 1 non-empty document required. "
                "Series contains no valid text."
            )

        if sample_n is not None and sample_n < len(clean):
            clean = clean.sample(n=sample_n, random_state=42)

        results: list[SentimentResult] = []

        for idx, text in enumerate(clean):
            tokens = [
                t for t in self._TOKEN_PATTERN.findall(text.lower())
                if len(t) >= 2
            ]
            raw_score, pos_count, neg_count = self._token_scorer.score(tokens)
            polarity = self._polarity_normalizer.normalize(raw_score, len(tokens))
            subjectivity = self._subjectivity_estimator.estimate(
                pos_count, neg_count, len(tokens)
            )
            label = self._label_assigner.assign(polarity, subjectivity)

            results.append(
                SentimentResult(
                    document_index=idx,
                    polarity_score=polarity,
                    subjectivity_score=subjectivity,
                    label=label,
                    positive_word_count=pos_count,
                    negative_word_count=neg_count,
                    word_count=len(tokens),
                )
            )

        label_counts: dict[str, int] = {}
        for r in results:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1

        polarities = [r.polarity_score for r in results]
        subjectivities = [r.subjectivity_score for r in results]

        return {
            "documents": [
                {
                    "document_index": r.document_index,
                    "polarity_score": r.polarity_score,
                    "subjectivity_score": r.subjectivity_score,
                    "label": r.label,
                    "positive_word_count": r.positive_word_count,
                    "negative_word_count": r.negative_word_count,
                    "word_count": r.word_count,
                }
                for r in results
            ],
            "corpus_summary": {
                "n_documents": len(results),
                "label_distribution": label_counts,
                "mean_polarity": round(sum(polarities) / len(polarities), 4),
                "mean_subjectivity": round(sum(subjectivities) / len(subjectivities), 4),
                "most_common_label": max(label_counts, key=label_counts.get),
            },
        }
