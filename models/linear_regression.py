"""Module for linear regression models"""
from abc import ABC , abstractmethod
import numpy as np
import pandas as pd
from evaluation.score import Score
from algorithms.optimizers.gradient_descent import GradientDescent


class BaseLinearRegressionModel(ABC):
    """Abstract base class for linear regression models"""
    @abstractmethod
    def fit(self ,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series] ,
     y : np.ndarray | pd.DataFrame[str] | list[pd.Series],
     **kwargs
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


class AnalyticalLinearRegression(BaseLinearRegressionModel):
    """Linear regression model"""
    def __init__(self) -> None:
        self.slope_ : float = None
        self.intercept_ : float = None
        self._x : np.ndarray = None
        self._y : np.ndarray = None
        self._y_true : np.ndarray = None
        self._y_pred : np.ndarray = None

    def fit(self, x: np.ndarray , y: np.ndarray , **kwargs):
        """fit de  analytical linear regresion"""
        slope = Slope().slope(x , y)
        intercept = Interception().interception(x , y , slope)
        self.slope_ = slope
        self.intercept_ = intercept
        self._x = x
        self._y_true = y

    def predict(self ,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series]
     ) -> np.ndarray:
        """Predict the y value"""
        if  self.intercept_ is None or self.slope_ is None :
            raise ValueError("The model must be Fitted before prediction")
        interception = self.intercept_
        slope = self.slope_
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

class AnalyticalMultipleLinearRegression(BaseLinearRegressionModel):
    """Multiple linear regression model"""
    def __init__(self) -> None:
        self._x_matrix : np.ndarray = None
        self._y_true : np.ndarray = None
        self._y_pred : np.ndarray = None
        self.coefficients_ : np.ndarray = None

    def fit (self , x : list[pd.Series] , y : pd.DataFrame[str] , **kwargs) -> None:
        """fit de  multiple linear regresion"""
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        coeffficients = OrdinaryLeastSquares().calculate_coefficients(x_matrix , y)
        self._y_true = y
        self.coefficients_ = coeffficients

    def predict(self ,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series]
     ) -> np.ndarray:
        """predict the y value"""
        if  self.coefficients_ is None :
            raise ValueError("The model Must be Fitted before prediction")
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        y_pred = x_matrix @ self.coefficients_
        self._y_pred = y_pred
        return y_pred

    def score(self,
     y_true : np.ndarray | pd.Series | list[pd.Series]
     )-> dict[str, float]:
        """calculate the score of the model"""
        return Score().get_score(y_true , self._y_pred)


class BaseGradientDescentLinearRegression(BaseLinearRegressionModel):
    """Base class for gradient descent linear regression models"""
    def __init__(self) -> None :
        self._x_matrix : np.ndarray = None
        self._y_true : np.ndarray = None
        self._y_pred : np.ndarray = None
        self.loss_history_ : list[float] = None
        self.iteration_ : int = None
        self.converged_ : bool = None
        self.coefficients_ : np.ndarray = None

    @abstractmethod
    def gradient_function(self,
     x : np.ndarray,
     y_true : np.ndarray,
     beta : np.ndarray
     ) -> np.ndarray:
        """Calculate the gradient of the loss function"""

    def fit(self , x : list[pd.Series] , y : pd.DataFrame[str] , **kwargs) -> None:
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        gradient_output = GradientDescent().optimize(self.gradient_function , x_matrix , y)
        self.coefficients_ = gradient_output[0]
        self.loss_history_ = gradient_output[1]
        self.iteration_ = gradient_output[2]
        self.converged_ = gradient_output[3]
        self._y_true = y

    def predict(self ,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series]
     ) -> np.ndarray:
        """predict the y value"""
        if self.coefficients_ is None :
            raise ValueError("The model must be fitted before prediction")
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        y_pred = x_matrix @ self.coefficients_
        self._y_pred = y_pred
        return y_pred

    def score(self,
     y_true : np.ndarray | pd.Series | list[pd.Series]
     )-> dict[str, float]:
        """calculate the score of the model"""
        return Score().get_score(y_true , self._y_pred)


class GradientDescentLinearRegression(BaseGradientDescentLinearRegression):
    """Gradient descent linear regression model"""
    def gradient_function(self,
     x : np.ndarray,
     y_true : np.ndarray,
     beta : np.ndarray
     ) -> np.ndarray:
        """Calculate the gradient of the loss function"""
        y_pred = x @ beta
        gradient = np.mean((y_pred - y_true)* x , axis = 0)
        return gradient


class GradientDescentMultipleLinearRegression(BaseGradientDescentLinearRegression):
    """Multiple Linear regresion using gradient descent"""
    def gradient_function(self,
     x : np.ndarray | pd.DataFrame[str] | list[pd.Series] ,
     y_true : np.ndarray | pd.DataFrame[str] | list[pd.Series] ,
     beta : np.ndarray
     ) -> np.ndarray:
        """Calculate the gradient of the loss function"""
        y_pred = x @ beta
        gradient = x.T @ (y_pred - y_true) / (len(y_true))
        return gradient

class LinearRegressionFactory:
    """Factory for linear regression models"""
    _registry : dict[ str , tuple[str , str , type[BaseLinearRegressionModel]]] = {}

    @classmethod
    def register(cls ,
        type_of_prediction : str ,
        complexity : str ,
        model : BaseLinearRegressionModel
        ) -> None:
        """Register a new linear regression model"""
        cls._registry[type_of_prediction , complexity] = model

    @classmethod
    def create(cls ,
        type_of_prediction : str
        , complexity : str
        ) -> BaseLinearRegressionModel:
        """Create a linear regression model"""
        model_data = cls._registry.get(type_of_prediction , complexity)
        if model_data is None:
            raise ValueError(
                f"Linear regression model {type_of_prediction , complexity} not registered"
                )
        model = model_data
        return model()


LinearRegressionFactory().register("gradient_descent"
                                ,  "simple"
                                ,  GradientDescentLinearRegression)
LinearRegressionFactory().register("gradient_descent"
                                ,  "multiple"
                                ,  GradientDescentMultipleLinearRegression)
LinearRegressionFactory().register("ordinary_least_squares"
                                ,  "simple"
                                ,  AnalyticalLinearRegression)
LinearRegressionFactory().register("ordinary_least_squares"
                                ,  "multiple"
                                ,  AnalyticalMultipleLinearRegression)

class LinearRegression(BaseLinearRegressionModel):
    """Dependency Injection for linear regression"""
    def __init__ (self,
        type_of_prediction : str = "gradient_descent" ,
        complexity : str = "simple"
        ) -> None:
        self.model = LinearRegressionFactory().create(
            type_of_prediction
            , complexity
            )

    def fit(self ,
        x : list[pd.Series] , y : pd.DataFrame[str] ,
        **kwargs
        ) -> None:
        """fit de  linear regresion"""
        fit_arguments = {
            "x" : x,
            "y" : y
        }
        if "batch_size" in kwargs:
            fit_arguments["batch_size"] = kwargs["batch_size"]

        if "initial_beta" in kwargs:
            fit_arguments["initial_beta"] = kwargs["initial_beta"]

        if "learning_rate" in kwargs:
            fit_arguments["learning_rate"] = kwargs["learning_rate"]

        if "max_iterations" in kwargs:
            fit_arguments["max_iterations"] = kwargs["max_iterations"]

        if "tolerance" in kwargs:
            fit_arguments["tolerance"] = kwargs["tolerance"]

        self._model.fit(**fit_arguments)

    def predict(self , x : np.ndarray | pd.DataFrame[str] | list[pd.Series]) -> np.ndarray:
        """predict the y value"""
        return self._model.predict(x)

    def score(self , y_true : np.ndarray | pd.Series | list[pd.Series]) -> dict[str,float]:
        """calculate the score of the model"""
        return self._model.score(y_true)



