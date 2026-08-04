"""Local embeddings via fastembed (ONNX, CPU, no API key).

`BAAI/bge-small-en-v1.5` produces 384-dimension vectors, which is what
`data_contexts.embedding` is declared as. Changing the model means changing that
column, so the dimension is asserted on first use rather than discovered when a
query silently returns nothing.

The model (~130MB) downloads once into the cache directory. That download is the
only network access this module ever performs; inference is local.
"""

from __future__ import annotations

import threading
from typing import Any

from lumen.embeddings.base import EmbeddingProvider

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_DIMENSIONS = 384


class FastEmbedProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        dimensions: int = _DEFAULT_DIMENSIONS,
        cache_dir: str | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._cache_dir = cache_dir
        self._embedder: Any | None = None
        # The first call triggers a model download; two concurrent worker jobs
        # must not race to write the same cache directory.
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
        if self._embedder is not None:
            return self._embedder

        with self._lock:
            if self._embedder is not None:
                return self._embedder
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - depends on install extras
                raise ImportError(
                    "Embeddings require fastembed. Install it with: "
                    "uv sync --directory engine --extra embeddings, "
                    "or set EMBEDDING_PROVIDER=none to disable semantic search."
                ) from exc

            self._embedder = TextEmbedding(
                model_name=self.model,
                cache_dir=self._cache_dir,
            )
            return self._embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embedder = self._ensure_loaded()
        vectors = [list(map(float, vector)) for vector in embedder.embed(texts)]

        actual = len(vectors[0])
        if actual != self.dimensions:
            raise ValueError(
                f"{self.model} produced {actual}-dimension vectors but the schema expects "
                f"{self.dimensions}. Update EMBEDDING_DIMENSIONS and migrate "
                "data_contexts.embedding to vector({actual}) before using this model."
            )
        return vectors
