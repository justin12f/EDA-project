"""Module for linear regression models"""
from abc import ABC , abstractmethod
import numpy as np
import pandas as pd
from evaluation.score import Score


class BaseLinearRegressionModel(ABC):
    """Abstract base class for linear regression models"""
    @abstractmethod
    def fit(self ,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series] ,
     y : np.ndarray | pd.DataFrame[str] | list[pd.Series]
     ) -> None:
        """Fit the model to the data"""

    @abstractmethod
    def predict(self,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series]
     ) -> np.ndarray:
        """Predict the y value"""

    @abstractmethod
    def score(self,
     y_true : np.ndarray | pd.Series | list[pd.Series]
     ) -> dict[str , float]:
        """Calculate the score of the model"""

class Slope:
    """Calculate the slope of the linear regression model"""
    def slope(self, x : np.ndarray , y : np.ndarray) -> float:
        """calculate the earring value"""
        numerator = np.sum((x - x.mean()) * (y - y.mean()))
        denominator = np.sum((x - x.mean()) ** 2)
        beta1 = numerator / denominator
        return beta1

class Interception:
    """Calculate the interception of the linear regression model"""
    def interception(self, x: np.ndarray, y: np.ndarray , slope: float) -> float:
        """calculate the interception value"""
        beta0 = y.mean() - slope * x.mean()
        return beta0


class AnalyticalLinearRegresion(BaseLinearRegressionModel):
    """Linear regression model"""
    def __init__(self) -> None:
        self._slope : float = None
        self._intercept : float = None
        self._x : np.ndarray = None
        self._y : np.ndarray = None
        self._y_true : np.ndarray = None
        self._y_pred : np.ndarray = None

    def fit(self, x: np.ndarray , y: np.ndarray):
        """fit de  analytical linear regresion"""
        slope = Slope().slope(x , y)
        intercept = Interception().interception(x , y , slope)
        self._slope = slope
        self._intercept = intercept
        self._x = x
        self._y_true = y

    def predict(self ,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series]
     ) -> np.ndarray:
        """Predict the y value"""
        if  self._intercept is None or self._slope is None :
            raise ValueError("The model must be Fitted before prediction")
        interception = self._intercept
        slope = self._slope
        y_pred = interception + (slope * x )
        self._y_pred = y_pred
        return y_pred

    def score(self,
     y_true : np.ndarray | pd.Series | list[pd.Series]
     ) -> dict[str,float]:
        """calculate the score of the model"""
        return Score().get_score(y_true , self._y_pred)



class BuildDesignMatrix:
    """Build the X matrix for the  multiple linear regression model"""
    def build_design_matrix (self , columns : list[pd.Series]):
        """build the deign matrix"""
        column_shape = columns[0].shape[0]
        initial_matrix = np.ones((column_shape , 1))
        x_matrix = initial_matrix
        for column in columns:
            x_matrix = np.hstack((x_matrix , column.values.reshape(-1 , 1)))
        return x_matrix

class OrdinaryLeastSquares:
    """
    Calculate the coefficients of the linear regression model
    using the ordinary least squares method
    """
    def calculate_coefficients(self , x : np.ndarray , y : np.ndarray) -> np.ndarray:
        """calculate the coefficients value"""
        beta = np.linalg.solve(x.T @ x , x.T @ y)
        return beta

class MultipleLinearRegression(BaseLinearRegressionModel):
    """Multiple linear regression model"""
    def __init__(self) -> None:
        self._x_matrix : np.ndarray = None
        self._y_true : np.ndarray = None
        self._y_pred : np.ndarray = None
        self._coefficients : np.ndarray = None

    def fit (self , x : list[pd.Series] , y : pd.DataFrame[str]) -> None:
        """fit de  multiple linear regresion"""
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        coeffficients = OrdinaryLeastSquares().calculate_coefficients(x_matrix , y)
        self._y_true = y
        self._coefficients = coeffficients

    def predict(self ,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series]
     ) -> np.ndarray:
        """predict the y value"""
        if  self._coefficients is None :
            raise ValueError("The model Must be Fitted before prediction")
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        y_pred = x_matrix @ self._coefficients
        self._y_pred = y_pred
        return y_pred

    def score(self,
     y_true : np.ndarray | pd.Series | list[pd.Series]
     )-> dict[str, float]:
        """calculate the score of the model"""
        return Score().get_score(y_true , self._y_pred)
















    