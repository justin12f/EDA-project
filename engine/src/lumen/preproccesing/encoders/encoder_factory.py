"""Encoder factory with multi-backend support.

Usage:
    # Polars backend (default)
    encoder = Encoder("one_hot")
    encoder.fit(polars_df)
    result = encoder.transform()

    # PySpark backend
    encoder = Encoder("one_hot", backend="spark")
    encoder.fit(spark_df)
    result = encoder.transform()
"""

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `EncoderFactory` + `EncoderInyeccionDependency` (polars | spark | pandas) desde Factory Maestra de Agentes.
# - ABSTRACCIÓN DEL DATO: `fit`/`transform` reciben el frame del backend registrado en `(encoder_type, backend)`.
# - REFACTOR NATIVO: Verificar todas las implementaciones registradas; corregir factory si `create()` falla en algún backend.
# #[AI_CONTEXT_END]
from __future__ import annotations

from typing import Any

from lumen.core.backend import Backend, DEFAULT_BACKEND
from lumen.preproccesing.encoders.implementations.base import AbstractEncoder

# ── Polars encoders ───────────────────────────────────────────────────────────
from lumen.preproccesing.encoders.polars_impl import (
    PolarsOneHotEncoder,
    PolarsOrdinalEncoder,
)

# ── Spark encoders ────────────────────────────────────────────────────────────
from lumen.preproccesing.encoders.spark_impl import (
    SparkOneHotEncoder,
    SparkOrdinalEncoder,
)

from lumen.preproccesing.encoders.pandas_impl import (
    PandasOneHotEncoder,
    PandasOrdinalEncoder,
)

class EncoderFactory:
    """Factory for creating encoders with multi-backend support.

    Registry maps ``(encoder_type, backend)`` → ``EncoderClass``.
    """

    _registry: dict[tuple[str, str], type[AbstractEncoder]] = {}

    @classmethod
    def register(
        cls,
        encoder_type: str,
        backend: str,
        encoder: type[AbstractEncoder],
    ) -> None:
        """Register an encoder under a (type, backend) key."""
        cls._registry[(encoder_type, backend)] = encoder

    @classmethod
    def create(
        cls,
        encoder_class: str,
        backend: Backend = "polars",
    ) -> AbstractEncoder:
        """Create an encoder for the given type and backend.

        Args:
            encoder_class: Encoder type name (e.g., ``"one_hot"``).
            backend: ``"polars"`` or ``"spark"``.

        Returns:
            An ``AbstractEncoder`` instance.

        Raises:
            ValueError: If the (type, backend) pair is not registered.
        """
        key = (encoder_class, backend)
        encoder_cls = cls._registry.get(key)
        if encoder_cls is None:
            available = [
                f"{t} ({b})" for (t, b) in sorted(cls._registry.keys())
            ]
            raise ValueError(
                f"Encoder '{encoder_class}' for backend '{backend}' not found. "
                f"Available: {available}"
            )
        return encoder_cls()

# ═══════════════════════════════════════════════════════════════════════════════
# Register all built-in encoders
# ═══════════════════════════════════════════════════════════════════════════════

# ── Polars ────────────────────────────────────────────────────────────────────
EncoderFactory.register("one_hot", "polars", PolarsOneHotEncoder)
EncoderFactory.register("ordinal", "polars", PolarsOrdinalEncoder)

# ── PySpark ───────────────────────────────────────────────────────────────────
EncoderFactory.register("one_hot", "spark", SparkOneHotEncoder)
EncoderFactory.register("ordinal", "spark", SparkOrdinalEncoder)

# ── Pandas ────────────────────────────────────────────────────────────────────
EncoderFactory.register("one_hot", "pandas", PandasOneHotEncoder)
EncoderFactory.register("ordinal", "pandas", PandasOrdinalEncoder)

class Encoder:
    """Dependency injection wrapper for encoders.

    Provides a unified API regardless of backend.

    Args:
        encoder: Encoder type name.
        backend: Backend engine — ``"polars"`` or ``"spark"``.
    """

    def __init__(self, encoder: str, backend: Backend = DEFAULT_BACKEND) -> None:
        self.encoder = EncoderFactory.create(encoder, backend)

    def fit(self, data: Any, **kwargs: Any) -> None:
        """Fit the encoder to the data."""
        self.encoder.fit(data, **kwargs)

    def transform(self) -> Any:
        """Transform the data."""
        return self.encoder.transform()
