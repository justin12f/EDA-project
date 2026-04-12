"""Module to create the gradient descent algorithm"""
from __future__ import annotations
from typing import Callable
import numpy as np 
import pandas as pd



class VerifyConvergence:
    """"Verify the converse of the algorithm"""
    def verify_convergence(self,
        loss_history : list[float],
        tolerance : float,
        iteration : int
        ) -> bool:
        """Verify the convergence of the algorithm"""
        if iteration == 0:
            return False

        if abs(loss_history[iteration] - loss_history[iteration-1]) < tolerance:
            return True

        return False

def loss_function (
    x : np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray] ,
    y :  np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray],
    beta : np.ndarray
    ) -> np.ndarray:
    """Calculate the loss of the model"""

    loss = np.sum((x @ beta - y )**2) / (2*len(y))
    return loss


class BatchSelector:
    """Select the batch of data for gradient descent"""
    def select(self,
        x          : np.ndarray,
        y          : np.ndarray,
        batch_size : int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select x and y batch based on batch_size"""
        #Clasic Descent
        if batch_size is None:
            return x, y

        #Stocastic Gradient Descent
        if batch_size == 1:
            index = np.random.randint(0, len(x))
            return x[[index]], y[[index]]

        #Mini Batch Gradient Descent
        index = np.random.choice(len(x), batch_size, replace=False)

        x = x[index]
        y = y[index]

        return x, y

class GradientDescentOptimizer:
    """Optimizer for gradient descent algorithm"""
    def optimize(self,
        gradient_function : Callable ,
        x_batch : np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray],
        y_batch : np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray],
        initial_beta : np.ndarray,
        batch_size : int = None,
        learning_rate : float = 0.01,
        max_iterations : int = 1000,
        tolerance : float = 0.00001
        ) -> np.ndarray:
        """Optimize the parameters of a model using gradient descent"""

        loss_history = []
        beta = initial_beta
        converged_ = False

        for iteration in range(max_iterations):

            #Select the batch of data
            x_batch , y_batch = BatchSelector().select(x_batch , y_batch , batch_size)

            #Calculate the loss
            loss = loss_function(x_batch, y_batch, beta)
            loss_history.append(loss)

            #Verify the convergence
            if VerifyConvergence().verify_convergence(loss_history , tolerance , iteration):
                converged_ = True
                break

            #Calculate the gradient
            gradient = gradient_function(x_batch, y_batch  , beta)

            #Update the parameters
            beta = beta - learning_rate * gradient

        loss_history_ = loss_history
        iteration_ = iteration

        return (beta , loss_history_ , iteration_ , converged_)

class GradientDescent:
    """Algorithm for optimize models"""
    def __init__(self,
        learning_rate : float = 0.01 ,
        max_iterations : int = 1000 ,
        tolerance : float = 1e-6
        ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self,
        gradient_function : Callable ,
        x : np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray],
        y : np.ndarray | pd.Series | list[pd.Series] | list[np.ndarray],
        initial_beta : np.ndarray = None
        ) -> tuple[np.ndarray , list[float], int]:
        """Optimize the parameters of a model using gradient descent"""

        if initial_beta is None:
            initial_beta = np.zeros(x.shape[1])

        learning_rate = self.learning_rate
        max_iterations = self.max_iterations
        tolerance = self.tolerance

        return GradientDescentOptimizer().optimize(
            gradient_function = gradient_function,
            x_batch = x,
            y_batch = y,
            initial_beta = initial_beta,
            learning_rate = learning_rate,
            max_iterations = max_iterations,
            tolerance = tolerance
        )



