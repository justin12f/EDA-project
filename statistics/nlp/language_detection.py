"""Character n-gram language detection for common languages."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LanguageProfile:
    """Immutable language profile for scoring."""

    language_code: str
    language_name: str
    trigram_set: frozenset[str]


class TrigramExtractor:
    """Extracts character trigrams from text for language fingerprinting.

    Character trigrams are the standard signal for language identification
    (Cavnar & Trenkle, 1994). They are robust to noise, short texts,
    and unknown words.
    """

    _CLEANUP_PATTERN: re.Pattern = re.compile(r"[^a-záéíóúüñàèìòùâêîôûäëïöüœæßçøå\s]",
                                               re.UNICODE)

    def extract(self, text: str, max_trigrams: int = 300) -> set[str]:
        """Extract unique character trigrams from text.

        Args:
            text: Raw document string.
            max_trigrams: Maximum unique trigrams to extract.

        Returns:
            Set of trigram strings.
        """
        clean = self._CLEANUP_PATTERN.sub("", text.lower())
        padded = f" {clean} "
        trigrams: set[str] = set()
        for i in range(len(padded) - 2):
            trigrams.add(padded[i:i + 3])
            if len(trigrams) >= max_trigrams:
                break
        return trigrams


class LanguageProfileLibrary:
    """Built-in character trigram profiles for 7 languages.

    Profiles derived from the most discriminative trigrams for each
    language. Covers: English, Spanish, French, German, Portuguese,
    Italian, Dutch.
    """

    _PROFILES: list[LanguageProfile] = [
        LanguageProfile(
            language_code="en",
            language_name="English",
            trigram_set=frozenset({
                "the", "ing", "and", "ion", "ent", "tio", "her", "ati",
                "for", "ter", "hat", "his", "ere", "con", "res", "ver",
                "all", "ons", "nce", "men", "ith", "ted", "ers", "pro",
                "not", "est", "rea", "ect", "has", "tha", "thi", "ome",
                "was", "are", "ate", "ble", "had", "our", "can", "tic",
                "but", "der", "ove", "int", "mor", "wit", "from", "que",
            }),
        ),
        LanguageProfile(
            language_code="es",
            language_name="Spanish",
            trigram_set=frozenset({
                "que", "nte", "con", "est", "ent", "los",
                "las", "par", "aci", "com", "ción", "una", "del", "por",
                "ado", "nto", "tra", "ara", "pro", "ida", "mos", "cia",
                "pre", "res", "era", "ero", "ién", "and",
                "ión", "ues", "dor", "pri", "nal", "mor", "tri", "ist",
                "rio", "men", "tro", "uer", "ría", "ene", "has",
            }),
        ),
        LanguageProfile(
            language_code="fr",
            language_name="French",
            trigram_set=frozenset({
                "les", "ent", "que", "des", "ion", "ons", "est", "une",
                "tio", "ati", "men", "eur", "aut", "tre", "ire", "sur",
                "par", "con", "ant", "our", "ais", "uit", "ont", "aux",
                "ier", "peu", "ans", "pas", "ils", "ous", "qui", "cet",
                "tte", "pou", "sse",  "ère", "nce", "ait", "lle",
                "eau", "rre", "ble", "âme", "oui", "côt", "dès", "fai",
            }),
        ),
        LanguageProfile(
            language_code="de",
            language_name="German",
            trigram_set=frozenset({
                "der", "die", "und", "ein", "ung", "sch", "ich", "den",
                "ist", "cht", "mit", "ver", "sie", "ste", "ren",
                "gen", "ter", "auf", "ber", "ner", "des", "wie", "das",
                "aus", "hen", "ten", "hat", "dem", "war", "bei", "lic",
                "men", "vor", "hin", "zur", "als", "nge", "nis", "sen",
                "iel", "tat", "sse", "nch", "end", "ger", "kei", "eit",
            }),
        ),
        LanguageProfile(
            language_code="pt",
            language_name="Portuguese",
            trigram_set=frozenset({
                "que", "ção", "ent", "com", "est", "par",
                "dos", "nte", "ões", "uma", "por", "mos", "pro", "tra",
                "res", "são", "não", "era", "ado", "ais", "mes", "ema",
                "ber", "ina", "ria", "men", "uer", "iss", "iva",
                "cer", "tro", "tas", "tos", "ita", "ote", "lho", "ãos",
                "pré", "ens", "ume", "nco", "cal", "tes",
            }),
        ),
        LanguageProfile(
            language_code="it",
            language_name="Italian",
            trigram_set=frozenset({
                "che", "ion", "ent", "del", "con", "zione", "per", "ell",
                "dei", "gli", "una", "lla", "non", "est", "ato", "oni",
                "tra", "pre", "pro", "nte", "era", "ici", "ore",
                "ite", "ari", "ati", "esi", "osa", "amo", "are", "all",
                "ene", "dde", "abb", "ssa", "ove", "uni", "mpo",
                "tti", "ste", "lle", "ndo", "ves", "ppe", "bbe",
            }),
        ),
        LanguageProfile(
            language_code="nl",
            language_name="Dutch",
            trigram_set=frozenset({
                "van", "het", "een", "ing", "oor", "ver", "aan", "sch",
                "den", "aar", "erd", "gen", "ste", "nde", "tig",
                "ken", "ren", "ter", "bij", "niet", "voor", "zijn",
                "worden", "maar", "heeft", "ook", "dat", "met",
                "dit", "die", "zich", "haar", "hem", "uit",
                "als", "dan", "kan", "meer", "nog", "wel", "wij", "hoe",
            }),
        ),
    ]

    def all_profiles(self) -> list[LanguageProfile]:
        """Return all built-in language profiles."""
        return self._PROFILES


class LanguageScorer:
    """Scores a document's trigrams against each language profile.

    Score = |doc_trigrams ∩ lang_trigrams| / |doc_trigrams ∪ lang_trigrams|
    (Jaccard similarity between document trigrams and language profile)
    """

    def score_all(
        self,
        doc_trigrams: set[str],
        profiles: list[LanguageProfile],
    ) -> list[tuple[str, str, float]]:
        """Score document against all profiles.

        Args:
            doc_trigrams: Trigram set extracted from the document.
            profiles: Language profiles to score against.

        Returns:
            List of (code, name, score) tuples sorted descending by score.
        """
        scored: list[tuple[str, str, float]] = []

        for profile in profiles:
            intersection = doc_trigrams & profile.trigram_set
            union = doc_trigrams | profile.trigram_set
            jaccard = len(intersection) / len(union) if union else 0.0
            scored.append((profile.language_code, profile.language_name, jaccard))

        return sorted(scored, key=lambda x: x[2], reverse=True)


class LanguageDetectionCalculator:
    """Character trigram-based language detection per document.

    Workflow:
        calculator = LanguageDetectionCalculator()
        result = calculator.calculate(
            series=df["user_comment"],
            top_n_candidates=3,   # optional
        )
    """

    _MINIMUM_TEXT_LENGTH: int = 20

    def __init__(self) -> None:
        self._extractor = TrigramExtractor()
        self._library = LanguageProfileLibrary()
        self._scorer = LanguageScorer()

    def calculate(
        self,
        series: pd.Series,
        top_n_candidates: int = 3,
    ) -> dict:
        """Detect language for each document in the series.

        Args:
            series: String Series of documents.
            top_n_candidates: Number of language candidates to return per doc.

        Returns:
            Dict with per-document detection results and corpus distribution.

        Raises:
            ValueError: If top_n_candidates is invalid or series is empty.
        """
        if top_n_candidates < 1:
            raise ValueError(
                f"top_n_candidates must be >= 1. Got {top_n_candidates}."
            )

        clean = series.dropna().astype(str)
        clean = clean[clean.str.strip().str.len() > 0]

        if len(clean) == 0:
            raise ValueError("Series contains no valid text documents.")

        profiles = self._library.all_profiles()
        document_results: list[dict] = []
        language_counts: dict[str, int] = {}

        for idx, text in enumerate(clean):
            if len(text.strip()) < self._MINIMUM_TEXT_LENGTH:
                document_results.append({
                    "document_index": idx,
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "candidates": [],
                    "note": "Text too short for reliable detection.",
                })
                language_counts["unknown"] = language_counts.get("unknown", 0) + 1
                continue

            trigrams = self._extractor.extract(text)
            scored = self._scorer.score_all(trigrams, profiles)

            best_code, best_name, best_score = scored[0]
            candidates = [
                {
                    "language_code": code,
                    "language_name": name,
                    "score": round(score, 4),
                }
                for code, name, score in scored[:top_n_candidates]
            ]

            document_results.append({
                "document_index": idx,
                "detected_language": best_code,
                "detected_language_name": best_name,
                "confidence": round(best_score, 4),
                "candidates": candidates,
            })
            language_counts[best_code] = language_counts.get(best_code, 0) + 1

        dominant_language = max(language_counts, key=language_counts.get)

        return {
            "documents": document_results,
            "corpus_summary": {
                "n_documents": len(clean),
                "language_distribution": language_counts,
                "dominant_language": dominant_language,
                "is_multilingual": len(
                    {r["detected_language"] for r in document_results
                     if r["detected_language"] != "unknown"}
                ) > 1,
            },
            "top_n_candidates": top_n_candidates,
        }
