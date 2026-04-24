"""Module for abstract classes"""

import pandas as pd

from .implementations.base import BaseEncoder
from .implementations.implementations import OneHotEncoder, OrdinalEncoder


class EncoderFactory:
    """Factory for creating encoders"""

    _registry: dict[str, type[BaseEncoder]] = {}

    @classmethod
    def register(cls, encoder_type: str, encoder: type[BaseEncoder]) -> None:
        """Register an encoder"""
        cls._registry[encoder_type] = encoder

    @classmethod
    def create(cls, encoder_class: str) -> BaseEncoder:
        """Create an encoder"""
        if encoder_class not in cls._registry:
            raise ValueError(f"Encoder type {encoder_class} not found")
        return cls._registry[encoder_class]()


EncoderFactory.register("one_hot", OneHotEncoder)
EncoderFactory.register("ordinal", OrdinalEncoder)


class Encoder:
    """Dependency inyection for all encoders"""

    def __init__(self, encoder: str) -> None:
        self.encoder = EncoderFactory.create(encoder)

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the encoder to the data"""
        self.encoder.fit(data, **kwargs)

    def transform(self) -> pd.Series | pd.DataFrame:
        """Transform the data"""
        return self.encoder.transform()
