"""Module to create the gradient descent algorithm"""
import numpy as np 
import pandas as pd
from typing import Callable


class GradientDescentOptimize:
    """Optimizer for gradient descent algorithm"""
    def optimize(self,
        gradient_function : Callable ,
        X : np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray],
        y : np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray],
        initial_beta : float,
        learning_rate : float,
        max_iterations : int) -> np.ndarray:
        """I AM TIRED I JUST WANT TO SLEEP :( it's all for today thank u if you read this By: Justin :)"""


    




