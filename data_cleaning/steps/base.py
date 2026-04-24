"""Module containing abstract base classes and step implementations for data cleaning."""

from abc import ABC, abstractmethod

import pandas as pd

# Base class


class BaseStep(ABC):
    """Base class for all pipeline steps."""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        self._data_frame = data_frame

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the DataFrame and return the cleaned version."""


# Auto-detection helpers
