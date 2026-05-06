"""Modules for create the base class and import data procces libraries 
for extract clases and use for the contract definition"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseScaler(ABC):
    """
    Base class for all scalers.
    """

    @abstractmethod
    def fit ( self , data : pd.Series ) -> 'BaseScaler':
        """fit the data for scaling 
        
        Args:
            data : pd.Series
                
        Returns:
            BaseScaler
                
        """


    @abstractmethod
    def transform( self , data : pd.Series ) -> pd.Series:
        """transform the data
        
        Args:
            data : pd.Series
                
        Returns:
            pd.Series
                
        """
