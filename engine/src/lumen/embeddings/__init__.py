from __future__ import annotations

from typing import Literal

from lumen.embeddings.base import EmbeddingProvider, NullEmbeddingProvider

ProviderName = Literal["fastembed", "none"]

__all__ = [
    "EmbeddingProvider",
    "NullEmbeddingProvider",
    "get_embedding_provider",
]


def get_embedding_provider(
    name: ProviderName = "fastembed",
    *,
    model: str = "BAAI/bge-small-en-v1.5",
    dimensions: int = 384,
    cache_dir: str | None = None,
) -> EmbeddingProvider:
    if name == "none":
        return NullEmbeddingProvider()

    from lumen.embeddings.fastembed_provider import FastEmbedProvider

    return FastEmbedProvider(model=model, dimensions=dimensions, cache_dir=cache_dir)
