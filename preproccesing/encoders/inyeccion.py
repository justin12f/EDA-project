"""Encoder dependency injection."""

from __future__ import annotations

from typing import Any

from core.backend import DEFAULT_BACKEND
from core.inyeccion import BackendInyeccionDependency
from preproccesing.encoders.encoder_factory import EncoderFactory


class EncoderInyeccionDependency(BackendInyeccionDependency):
    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        super().__init__(backend)

    def create(self, encoder_type: str) -> Any:
        return EncoderFactory.create(encoder_type, backend=self._backend)
