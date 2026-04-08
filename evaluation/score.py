"""Module to calculate the score of the linear regression model"""
import numpy as np
import pandas as pd


class MeanSquareError:
    """calculate the MSE"""
    def mean_square_error(self,
    y_true : np.ndarray | pd.Series | list[pd.Series] ,
    y_pred : np.ndarray | pd.Series | list[pd.Series]
    ) -> float:
        """calculate the MSE value"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

class RootMeanSquareError:
    """"Calculate the rooted MSE"""
    def root_mean_square_error(self,
    y_true : np.ndarray | pd.Series | list[pd.Series] ,
    y_pred : np.ndarray | pd.Series | list[pd.Series]
    ) -> float:
        """calculate the root mininum square error value"""
        mse = MeanSquareError().mean_square_error(y_true , y_pred)
        return np.sqrt(mse)

class SquaredR:
    """Calculate the R**2 value"""
    def squared_r(self,
    y_true : np.ndarray | pd.Series | list[pd.Series] ,
    y_pred : np.ndarray | pd.Series | list[pd.Series]
    ) -> float:
        """calculate the R**2 value"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true_mean = y_true.mean()
        numerator = np.sum((y_pred - y_true)**2)
        denominator = np.sum((y_true - y_true_mean)**2)
        return 1 - (numerator / denominator)

class Score:
    """Calculate the score of the linear regression model"""
    def get_score(self,
    y_true : np.ndarray | pd.Series | list[pd.Series] ,
    y_pred : np.ndarray | pd.Series | list[pd.Series]
    ) -> dict[str , float]:
        """calculates all the metrics"""
        mean_square_error = MeanSquareError().mean_square_error(y_true , y_pred)
        root_mean_square_error = RootMeanSquareError().root_mean_square_error(y_true , y_pred)
        squared_r = SquaredR().squared_r(y_true , y_pred)
        return {
            "mean_square_error" : mean_square_error,
            "root_mean_square_error" : root_mean_square_error,
            "squared_r" : squared_r
        }
