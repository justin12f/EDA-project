""" Module for convert the data """

import  pandas as pd
import numpy as np
from preproccesing.scalers.base import BaseScaler

class StandardScaler(BaseScaler):
    """standarization between -1 and 1
    """

    mean_ : float
    std_ : float

    def fit( self , data : pd.Series )-> None:
        """
        fit method for fit the data in the class StandarScaler
        Args:
            data (pd.Series): ->
        """
        self.mean_ = np.mean(data)
        self.std_ = np.std(data)

    def transform( self , data : pd.Series ) -> pd.Series:
        """
        transform method for transform the data in the class StandarScaler
        Args:
            data (pd.Series): ->
        """
        standar_data = ( data - self.mean_ ) / self.std_
        return standar_data

    def fit_transform( self, data: pd.Series) -> pd.Series:
        """
        fit_transform method for fit and transform the data in the class StandarScaler
        Args:
            data (pd.Series): ->
        """
        self.fit(data)
        return self.transform(data)
