"""Module for linear regression models"""
from functools import cached_property
import numpy as np
import pandas as pd

class LinearRegresion :
    """Linear regression model"""
    def __init__(self , x : list[float] | np.array | pd.Series , y : list[float] | np.array | pd.Series) -> None:
        self._x = np.array(x)
        self._y = np.array(y)

    @cached_property
    def _slope(self) -> float :
        """calculate the earring value"""
        x = self._x
        y = self._y
        numerator = np.sum((x - x.mean())* ( y - y.mean()))
        denominator =  np.sum(x - x.mean())**2
        beta1 = numerator / denominator
        return beta1

    @cached_property
    def _interception(self) -> float :
        """calculate the interception value"""
        x = self._x
        y = self._y
        beta0 = y.mean() - self._slope() * x.mean()
        return beta0

    @cached_property
    def mean_squared_error(self) -> float :
        """calculate the error value"""
        y_pred = self._slope * self._x + self._interception
        return np.mean((self._y - y_pred)**2)

    def predict(self , x : list[float] | np.array | pd.Series) -> np.array :
        """predict the values"""
        x = np.array(x)
        return self._slope * x + self._interception





