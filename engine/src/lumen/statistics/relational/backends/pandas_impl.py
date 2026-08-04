"""Pandas adapter backend for the relational statistics domain."""
from __future__ import annotations

from typing import Any
import pandas as pd

from relational.abstract.contingency_analysis import AbstractContingencyAnalysisCalculator
from relational.abstract.correlation_matrix import AbstractCorrelationMatrixCalculator
from relational.abstract.cross_correlation import AbstractCrossCorrelationCalculator
from relational.abstract.granger_causality import AbstractGrangerCausalityCalculator
from relational.abstract.interaction_effects import AbstractInteractionEffectsCalculator
from relational.abstract.multicollinearity import AbstractMulticollinearityCalculator
from relational.abstract.mutual_information import AbstractMutualInformationCalculator
from relational.abstract.partial_correlation import AbstractPartialCorrelationCalculator

from relational.contingency_analysis import ContingencyAnalysisCalculator
from relational.correlation_matrix import CorrelationMatrixCalculator
from relational.cross_correlation import CrossCorrelationCalculator
from relational.granger_causality import GrangerCausalityCalculator
from relational.interaction_effects import InteractionEffectsCalculator
from relational.multicollinearity import MulticollinearityCalculator
from relational.mutual_information import MutualInformationCalculator
from relational.partial_correlation import PartialCorrelationCalculator


class ContingencyAnalysisCalculatorPandas(AbstractContingencyAnalysisCalculator):
    def __init__(self) -> None:
        self._impl = ContingencyAnalysisCalculator()

    def calculate(
        self,
        data: Any,
        col1: str,
        col2: str,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, col1, col2)


class CorrelationMatrixCalculatorPandas(AbstractCorrelationMatrixCalculator):
    def __init__(self) -> None:
        self._impl = CorrelationMatrixCalculator()

    def calculate(
        self,
        data: Any,
        columns: list[str],
        method: str = "pearson",
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, columns, method)


class CrossCorrelationCalculatorPandas(AbstractCrossCorrelationCalculator):
    def __init__(self) -> None:
        self._impl = CrossCorrelationCalculator()

    def calculate(
        self,
        data: Any,
        col1: str,
        col2: str,
        max_lag: int = 10,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, col1, col2, max_lag)


class GrangerCausalityCalculatorPandas(AbstractGrangerCausalityCalculator):
    def __init__(self) -> None:
        self._impl = GrangerCausalityCalculator()

    def calculate(
        self,
        data: Any,
        target_column: str,
        predictor_column: str,
        max_lag: int = 5,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, target_column, predictor_column, max_lag)


class InteractionEffectsCalculatorPandas(AbstractInteractionEffectsCalculator):
    def __init__(self) -> None:
        self._impl = InteractionEffectsCalculator()

    def calculate(
        self,
        data: Any,
        target_column: str,
        features: list[str],
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, target_column, features)


class MulticollinearityCalculatorPandas(AbstractMulticollinearityCalculator):
    def __init__(self) -> None:
        self._impl = MulticollinearityCalculator()

    def calculate(
        self,
        data: Any,
        columns: list[str],
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, columns)


class MutualInformationCalculatorPandas(AbstractMutualInformationCalculator):
    def __init__(self) -> None:
        self._impl = MutualInformationCalculator()

    def calculate(
        self,
        data: Any,
        target_column: str,
        feature_columns: list[str],
        is_target_discrete: bool = True,
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, target_column, feature_columns, is_target_discrete)


class PartialCorrelationCalculatorPandas(AbstractPartialCorrelationCalculator):
    def __init__(self) -> None:
        self._impl = PartialCorrelationCalculator()

    def calculate(
        self,
        data: Any,
        col1: str,
        col2: str,
        covariates: list[str],
    ) -> dict[str, Any]:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        return self._impl.calculate(df, col1, col2, covariates)
