"""Module for abstract classes"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseEncoder(ABC):
    """Abstract class for all encoders"""

    @abstractmethod
    def fit(self, data: pd.Series | list[pd.Series], **kwargs) -> None:
        """Fit the encoder to the data"""

    @abstractmethod
    def transform(self) -> pd.Series | pd.DataFrame:
        """Transform the data"""


class BaseTransform(ABC):
    """Abstract class for all encoders"""

    @abstractmethod
    def transform(self) -> pd.Series | pd.DataFrame:
        """Transform the data"""
