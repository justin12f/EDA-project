"""Embedding contract.

Kept separate from `lumen.llm` on purpose: neither of the two configured chat
providers (Anthropic, Groq) offers embeddings, so this is a different vendor
axis with a different failure mode. Treating them as one interface would mean
every LLM swap risks silently changing the vector dimension, which is a
migration, not a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input, in the same order.

        Implementations must be deterministic for the same input and model:
        a stored vector and a query vector have to come from the same function
        or similarity is meaningless.
        """

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class NullEmbeddingProvider(EmbeddingProvider):
    """Explicitly disabled embeddings.

    Selected with `EMBEDDING_PROVIDER=none`. Context rows are still written —
    with a null embedding — so nothing is lost and a later backfill can embed
    them. Semantic search simply returns nothing until then, which is the
    honest behaviour: better an empty result than a silently wrong one.
    """

    model = "none"
    dimensions = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]
