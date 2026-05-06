"""Module for linear regression models"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from algorithms.optimizers.gradient_descent import GradientDescent
from evaluation.score import Score
from preproccesing.encoders.encoder_factory import Encoder


class BaseLinearRegressionModel(ABC):
    """Abstract base class for linear regression models"""

    @abstractmethod
    def fit(
        self,
        x: np.ndarray | pd.DataFrame | list[pd.Series],
        y: np.ndarray | pd.DataFrame | list[pd.Series],
        **kwargs,
    ) -> None:
        """Fit the model to the data"""

    @abstractmethod
    def predict(self, x: np.ndarray | pd.DataFrame | list[pd.Series]) -> np.ndarray:
        """Predict the y value"""

    @abstractmethod
    def score(
        self, y_true: np.ndarray | pd.Series | list[pd.Series]
    ) -> dict[str, float]:
        """Calculate the score of the model"""


class Slope:
    """Calculate the slope of the linear regression model"""

    def slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """calculate the earring value"""
        numerator = np.sum((x - x.mean()) * (y - y.mean()))
        denominator = np.sum((x - x.mean()) ** 2)
        beta1 = numerator / denominator
        return beta1


class Interception:
    """Calculate the interception of the linear regression model"""

    def interception(self, x: np.ndarray, y: np.ndarray, slope: float) -> float:
        """calculate the interception value"""
        beta0 = y.mean() - slope * x.mean()
        return beta0


class AnalyticalLinearRegression(BaseLinearRegressionModel):
    """Linear regression model"""

    def __init__(self) -> None:
        self.slope_: float = None
        self.intercept_: float = None
        self._x: np.ndarray = None
        self._y: np.ndarray = None
        self._y_true: np.ndarray = None
        self._y_pred: np.ndarray = None

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """fit de  analytical linear regresion"""
        slope = Slope().slope(x, y)
        intercept = Interception().interception(x, y, slope)
        self.slope_ = slope
        self.intercept_ = intercept
        self._x = x
        self._y_true = y

    def predict(self, x: np.ndarray | pd.DataFrame | list[pd.Series]) -> np.ndarray:
        """Predict the y value"""
        if self.intercept_ is None or self.slope_ is None:
            raise ValueError("The model must be Fitted before prediction")
        interception = self.intercept_
        slope = self.slope_
        y_pred = interception + (slope * x)
        self._y_pred = y_pred
        return y_pred

    def score(
        self, y_true: np.ndarray | pd.Series | list[pd.Series]
    ) -> dict[str, float]:
        """calculate the score of the model"""
        return Score().get_score(y_true, self._y_pred)


class BuildDesignMatrix:
    """Build the X matrix for the  multiple linear regression model"""

    def build_design_matrix(
        self, columns: list[pd.Series] | pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """build the design matrix"""
        if isinstance(columns, pd.DataFrame):
            columns_list = [columns[col] for col in columns.columns]
        elif isinstance(columns, pd.Series):
            columns_list = [columns]
        elif isinstance(columns, np.ndarray):
            if columns.ndim == 1:
                columns_list = [pd.Series(columns)]
            else:
                columns_list = [
                    pd.Series(columns[:, i]) for i in range(columns.shape[1])
                ]
        else:
            columns_list = columns

        column_shape = columns_list[0].shape[0]
        initial_matrix = np.ones((column_shape, 1))
        x_matrix = initial_matrix
        for column in columns_list:
            x_matrix = np.hstack((x_matrix, column.values.reshape(-1, 1)))
        return x_matrix


class OrdinaryLeastSquares:
    """
    Calculate the coefficients of the linear regression model
    using the ordinary least squares method
    """

    def calculate_coefficients(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """calculate the coefficients value"""
        beta = np.linalg.solve(x.T @ x, x.T @ y)
        return beta


class AnalyticalMultipleLinearRegression(BaseLinearRegressionModel):
    """Multiple linear regression model"""

    def __init__(self) -> None:
        self._x_matrix: np.ndarray = None
        self._y_true: np.ndarray = None
        self._y_pred: np.ndarray = None
        self.coefficients_: np.ndarray = None

    def fit(self, x: list[pd.Series], y: pd.DataFrame, **kwargs) -> None:
        """fit de  multiple linear regresion"""
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        coeffficients = OrdinaryLeastSquares().calculate_coefficients(x_matrix, y)
        self._y_true = y
        self.coefficients_ = coeffficients

    def predict(self, x: np.ndarray | pd.DataFrame | list[pd.Series]) -> np.ndarray:
        """predict the y value"""
        if self.coefficients_ is None:
            raise ValueError("The model Must be Fitted before prediction")
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        y_pred = x_matrix @ self.coefficients_
        self._y_pred = y_pred
        return y_pred

    def score(
        self, y_true: np.ndarray | pd.Series | list[pd.Series]
    ) -> dict[str, float]:
        """calculate the score of the model"""
        return Score().get_score(y_true, self._y_pred)


class BaseGradientDescentLinearRegression(BaseLinearRegressionModel):
    """Base class for gradient descent linear regression models"""

    def __init__(self) -> None:
        self._x_matrix: np.ndarray = None
        self._y_true: np.ndarray = None
        self._y_pred: np.ndarray = None
        self.loss_history_: list[float] = None
        self.iteration_: int = None
        self.converged_: bool = None
        self.coefficients_: np.ndarray = None

    @abstractmethod
    def gradient_function(
        self, x: np.ndarray, y_true: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Calculate the gradient of the loss function"""

    def fit(self, x: list[pd.Series], y: pd.DataFrame, **kwargs) -> None:
        x_matrix = BuildDesignMatrix().build_design_matrix(x)

        gd_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["learning_rate", "max_iterations", "tolerance"]
        }
        opt_kwargs = {
            k: v for k, v in kwargs.items() if k in ["initial_beta", "batch_size"]
        }

        gradient_output = GradientDescent(**gd_kwargs).optimize(
            self.gradient_function, x_matrix, y, **opt_kwargs
        )
        self.coefficients_ = gradient_output[0]
        self.loss_history_ = gradient_output[1]
        self.iteration_ = gradient_output[2]
        self.converged_ = gradient_output[3]
        self._y_true = y

    def predict(self, x: np.ndarray | pd.DataFrame | list[pd.Series]) -> np.ndarray:
        """predict the y value"""
        if self.coefficients_ is None:
            raise ValueError("The model must be fitted before prediction")
        x_matrix = BuildDesignMatrix().build_design_matrix(x)
        y_pred = x_matrix @ self.coefficients_
        self._y_pred = y_pred
        return y_pred

    def score(
        self, y_true: np.ndarray | pd.Series | list[pd.Series]
    ) -> dict[str, float]:
        """calculate the score of the model"""
        return Score().get_score(y_true, self._y_pred)


class GradientDescentLinearRegression(BaseGradientDescentLinearRegression):
    """Gradient descent linear regression model"""

    def gradient_function(
        self, x: np.ndarray, y_true: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Calculate the gradient of the loss function"""
        y_pred = x @ beta
        gradient = np.mean((y_pred - y_true) * x, axis=0)
        return gradient


class GradientDescentMultipleLinearRegression(BaseGradientDescentLinearRegression):
    """Multiple Linear regresion using gradient descent"""

    def gradient_function(
        self,
        x: np.ndarray | pd.DataFrame | list[pd.Series],
        y_true: np.ndarray | pd.DataFrame | list[pd.Series],
        beta: np.ndarray,
    ) -> np.ndarray:
        """Calculate the gradient of the loss function"""
        y_pred = x @ beta
        gradient = x.T @ (y_pred - y_true) / (len(y_true))
        return gradient


class LinearRegressionFactory:
    """Factory for linear regression models"""

    _registry: dict[str, tuple[str, str, type[BaseLinearRegressionModel]]] = {}

    @classmethod
    def register(
        cls, type_of_prediction: str, complexity: str, model: BaseLinearRegressionModel
    ) -> None:
        """Register a new linear regression model"""
        cls._registry[type_of_prediction, complexity] = model

    @classmethod
    def create(
        cls, type_of_prediction: str, complexity: str
    ) -> BaseLinearRegressionModel:
        """Create a linear regression model"""
        model_data = cls._registry.get((type_of_prediction, complexity))
        if model_data is None:
            raise ValueError(
                f"Linear regression model {type_of_prediction, complexity} not registered"
            )
        model = model_data
        return model()


LinearRegressionFactory().register(
    "gradient_descent", "simple", GradientDescentLinearRegression
)
LinearRegressionFactory().register(
    "gradient_descent", "multiple", GradientDescentMultipleLinearRegression
)
LinearRegressionFactory().register(
    "ordinary_least_squares", "simple", AnalyticalLinearRegression
)
LinearRegressionFactory().register(
    "ordinary_least_squares", "multiple", AnalyticalMultipleLinearRegression
)


class EncodeNeededValidation:
    """Validate the encoded data"""

    def __init__(
        self,
        data: pd.Series | pd.DataFrame | list[pd.Series],
        columns: list[str] = None,
    ) -> None:
        self.data = data
        self.columns = columns
        self.encoder = None

    def validate(self, type_of_encoder: str):
        """Validate the encoded data"""
        needs_encoding = False

        if isinstance(self.data, pd.Series):
            if pd.api.types.is_string_dtype(self.data) or pd.api.types.is_object_dtype(
                self.data
            ):
                needs_encoding = True
        elif isinstance(self.data, pd.DataFrame):
            if not self.data.select_dtypes(include=[object, "string"]).empty:
                needs_encoding = True
        elif isinstance(self.data, list):
            for col in self.data:
                if pd.api.types.is_string_dtype(col) or pd.api.types.is_object_dtype(
                    col
                ):
                    needs_encoding = True
                    break

        if needs_encoding:
            self.encoder = Encoder(type_of_encoder)

        return self.encoder


class AddToFitFromKwargs:
    """Add arguments to the fit method from kwargs"""

    def add_to_fit_from_kwargs(
        self,
        fit_arguments: dict[str, any],
        list_of_arguments: list[str],
        kwargs: dict[str, any],
    ) -> dict[str, any]:
        """Add arguments to the fit method from kwargs"""
        for argument in list_of_arguments:
            if argument in kwargs:
                fit_arguments.update({argument: kwargs[argument]})
        return fit_arguments


class LinearRegression(BaseLinearRegressionModel):
    """Dependency Injection for linear regression"""

    def __init__(
        self, type_of_prediction: str = "gradient_descent", complexity: str = "simple"
    ) -> None:
        self.model = LinearRegressionFactory().create(type_of_prediction, complexity)
        self.encoder = None

    def fit(self, x: list[pd.Series], y: pd.DataFrame, **kwargs) -> None:
        """fit de  linear regresion"""

        fit_arguments = {"x": x, "y": y}

        list_of_arguments = [
            "batch_size",
            "initial_beta",
            "learning_rate",
            "max_iterations",
            "tolerance",
            "type_of_encoder",
        ]

        fit_arguments = AddToFitFromKwargs().add_to_fit_from_kwargs(
            fit_arguments, list_of_arguments, kwargs
        )

        if "type_of_encoder" in fit_arguments:
            self.encoder = EncodeNeededValidation(x).validate(
                fit_arguments["type_of_encoder"]
            )
            if self.encoder is not None:
                x_df = x.to_frame() if isinstance(x, pd.Series) else pd.DataFrame(x)
                self.encoder.fit(x_df, **kwargs)
                encoded_df = self.encoder.transform()
                # For simplicity, if we end up with one column, extract it. Otherwise pass the dataframe
                x = (
                    encoded_df.iloc[:, 0]
                    if len(encoded_df.columns) == 1
                    else encoded_df
                )
        kwargs_to_pass = {
            k: v
            for k, v in fit_arguments.items()
            if k not in ["x", "y", "type_of_encoder"]
        }
        self.model.fit(x, y, **kwargs_to_pass)

    def predict(self, x: np.ndarray | pd.DataFrame | list[pd.Series]) -> np.ndarray:
        """predict the y value"""
        if self.encoder is not None:
            encoded_df = self.encoder.transform()
            x = encoded_df.iloc[:, 0] if len(encoded_df.columns) == 1 else encoded_df
        return self.model.predict(x)

    def score(
        self, y_true: np.ndarray | pd.Series | list[pd.Series]
    ) -> dict[str, float]:
        """calculate the score of the model"""
        return self.model.score(y_true)
