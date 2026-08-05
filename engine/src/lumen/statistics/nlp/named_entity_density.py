"""Named entity density estimation via rule-based pattern matching."""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en `NlpStatisticsFactory` (backends pandas | polars | spark) y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import pandas as pd

class EntityType(str, Enum):
    """Enumeration of detectable named entity types."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENTAGE = "PERCENTAGE"
    EMAIL = "EMAIL"
    URL = "URL"

@dataclass(frozen=True)
class EntityMatch:
    """Immutable record of a single named entity match."""

    entity_type: str
    matched_text: str
    start: int
    end: int

class EntityPatternLibrary:
    """Compiled regex patterns for rule-based named entity recognition.

    Pattern design principles:
        - High precision over recall (fewer false positives).
        - Each pattern is anchored to syntactic context where possible.
        - Ordering matters: more specific patterns should precede general ones.
    """

    PATTERNS: dict[str, re.Pattern] = {
        EntityType.EMAIL: re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
        ),
        EntityType.URL: re.compile(
            r"https?://[^\s<>\"{}|\\^\[\]`]+"
        ),
        EntityType.MONEY: re.compile(
            r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|thousand))?"
            r"|\d+(?:,\d{3})*(?:\.\d+)?\s?(?:USD|EUR|GBP|MXN|JPY|CAD)",
            re.IGNORECASE,
        ),
        EntityType.PERCENTAGE: re.compile(
            r"\d+(?:\.\d+)?\s?%"
        ),
        EntityType.DATE: re.compile(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?"
            r"(?:,?\s+\d{4})?"
            r"|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
            r"|\b\d{4}[/\-]\d{2}[/\-]\d{2}\b",
            re.IGNORECASE,
        ),
        EntityType.PERSON: re.compile(
            r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+"
            r"(?:\s+[A-Z][a-z]+)?"
            r"|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
        ),
        EntityType.ORGANIZATION: re.compile(
            r"\b[A-Z][A-Za-z&\s]+(?:Inc\.|Corp\.|LLC|Ltd\.|Co\.|Group|"
            r"University|Institute|Foundation|Association|Department|Ministry)"
            r"|\b(?:NASA|WHO|UN|EU|FBI|CIA|IMF|WTO|OPEC|NATO|UNICEF)"
            r"|\b[A-Z]{2,6}\b(?=\s+(?:Corp|Inc|Ltd|LLC|Group))",
            re.IGNORECASE,
        ),
        EntityType.LOCATION: re.compile(
            r"\b(?:in|at|from|near|to)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
            r"|\b[A-Z][a-z]+(?:,\s+[A-Z]{2})?\b(?=\s+(?:city|country|state|"
            r"province|region|district|county))",
            re.IGNORECASE,
        ),
    }

class DocumentEntityExtractor:
    """Extracts all named entity matches from a single document."""

    def __init__(self, pattern_library: EntityPatternLibrary) -> None:
        self._library = pattern_library

    def extract(self, text: str) -> list[EntityMatch]:
        """Find all entity matches in a document.

        Args:
            text: Raw document string.

        Returns:
            List of EntityMatch objects (may overlap across entity types).
        """
        matches: list[EntityMatch] = []
        for entity_type, pattern in self._library.PATTERNS.items():
            for match in pattern.finditer(text):
                matches.append(
                    EntityMatch(
                        entity_type=entity_type.value,
                        matched_text=match.group().strip(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return matches

class EntityDensityComputer:
    """Computes entity density as entities per 100 words."""

    def compute(self, entity_count: int, word_count: int) -> float:
        """Compute entity density.

        Args:
            entity_count: Total entities found in document.
            word_count: Total word count of the document.

        Returns:
            Entities per 100 words.
        """
        if word_count == 0:
            return 0.0
        return round(entity_count / word_count * 100, 4)

class NamedEntityDensityCalculator:
    """Rule-based named entity density analysis for a text column.

    Dependency-free alternative to spaCy/NLTK. Uses compiled
    regex patterns for EMAIL, URL, MONEY, PERCENTAGE, DATE,
    PERSON, ORGANIZATION, LOCATION.

    Workflow:
        calculator = NamedEntityDensityCalculator()
        result = calculator.calculate(
            series=df["article_body"],
            entity_types=None,    # optional, filters to specific types
            sample_n=None,        # optional
        )
    """

    _MINIMUM_DOCUMENTS: int = 1
    _WORD_PATTERN: re.Pattern = re.compile(r"\b\w+\b")

    def __init__(self) -> None:
        self._pattern_library = EntityPatternLibrary()
        self._extractor = DocumentEntityExtractor(self._pattern_library)
        self._density_computer = EntityDensityComputer()

    def calculate(
        self,
        series: pd.Series,
        entity_types: list[str] | None = None,
        sample_n: int | None = None,
    ) -> dict:
        """Detect and quantify named entities across all documents.

        Args:
            series: String Series of documents.
            entity_types: Subset of entity types to report. Defaults to all.
                Valid: 'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE',
                'MONEY', 'PERCENTAGE', 'EMAIL', 'URL'.
            sample_n: Optional sample size limit.

        Returns:
            Dict with per-document entity counts and corpus-level density stats.

        Raises:
            ValueError: If entity_types contains invalid type names.
        """
        _VALID_ENTITY_TYPES: frozenset[str] = frozenset(e.value for e in EntityType)

        if entity_types is not None:
            invalid = [t for t in entity_types if t not in _VALID_ENTITY_TYPES]
            if invalid:
                raise ValueError(
                    f"Invalid entity types: {invalid}. "
                    f"Valid: {sorted(_VALID_ENTITY_TYPES)}"
                )

        clean = series.dropna().astype(str)
        clean = clean[clean.str.strip().str.len() > 0]

        if len(clean) < self._MINIMUM_DOCUMENTS:
            raise ValueError("Series contains no valid text documents.")

        if sample_n is not None and sample_n < len(clean):
            clean = clean.sample(n=sample_n, random_state=42)

        document_results: list[dict] = []
        corpus_type_counts: dict[str, int] = {
            t.value: 0 for t in EntityType
        }
        total_words = 0

        for idx, text in enumerate(clean):
            word_count = len(self._WORD_PATTERN.findall(text))
            total_words += word_count
            all_matches = self._extractor.extract(text)

            if entity_types is not None:
                filtered_matches = [
                    m for m in all_matches if m.entity_type in entity_types
                ]
            else:
                filtered_matches = all_matches

            type_counts: dict[str, int] = {}
            for match in filtered_matches:
                type_counts[match.entity_type] = (
                    type_counts.get(match.entity_type, 0) + 1
                )
                corpus_type_counts[match.entity_type] = (
                    corpus_type_counts.get(match.entity_type, 0) + 1
                )

            total_entities = sum(type_counts.values())
            density = self._density_computer.compute(total_entities, word_count)

            document_results.append({
                "document_index": idx,
                "word_count": word_count,
                "total_entities": total_entities,
                "entity_density_per_100_words": density,
                "entity_type_counts": type_counts,
                "entities": [
                    {
                        "type": m.entity_type,
                        "text": m.matched_text,
                        "start": m.start,
                        "end": m.end,
                    }
                    for m in filtered_matches
                ],
            })

        total_entities_corpus = sum(corpus_type_counts.values())
        corpus_density = self._density_computer.compute(
            total_entities_corpus, total_words
        )

        return {
            "documents": document_results,
            "corpus_summary": {
                "n_documents": len(clean),
                "total_words": total_words,
                "total_entities": total_entities_corpus,
                "corpus_density_per_100_words": corpus_density,
                "entity_type_distribution": {
                    k: v for k, v in corpus_type_counts.items() if v > 0
                },
                "most_frequent_entity_type": (
                    max(corpus_type_counts, key=corpus_type_counts.get)
                    if total_entities_corpus > 0 else None
                ),
            },
            "entity_types_analysed": (
                entity_types if entity_types is not None
                else [t.value for t in EntityType]
            ),
        }
