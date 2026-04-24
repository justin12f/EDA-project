"""Module for data analysis"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseDataAnalysis(ABC):
    """base class for data analysis"""

    def __init__(self, data_frame: pd.DataFrame) -> None:
        self._data_frame = data_frame

    @abstractmethod
    def analyze(self, **kwargs) -> any:
        """analyze the data frame and return the results"""
