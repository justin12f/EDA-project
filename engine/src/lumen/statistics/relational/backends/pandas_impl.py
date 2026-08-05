"""Pandas adapter backend for the relational statistics domain."""
from __future__ import annotations

from typing import Any
import pandas as pd

from lumen.statistics.relational.abstract.contingency_analysis import AbstractContingencyAnalysisCalculator
from lumen.statistics.relational.abstract.correlation_matrix import AbstractCorrelationMatrixCalculator
from lumen.statistics.relational.abstract.cross_correlation import AbstractCrossCorrelationCalculator
from lumen.statistics.relational.abstract.granger_causality import AbstractGrangerCausalityCalculator
from lumen.statistics.relational.abstract.interaction_effects import AbstractInteractionEffectsCalculator
from lumen.statistics.relational.abstract.multicollinearity import AbstractMulticollinearityCalculator
from lumen.statistics.relational.abstract.mutual_information import AbstractMutualInformationCalculator
from lumen.statistics.relational.abstract.partial_correlation import AbstractPartialCorrelationCalculator

from lumen.statistics.relational.contingency_analysis import ContingencyAnalysisCalculator
from lumen.statistics.relational.correlation_matrix import CorrelationMatrixCalculator
from lumen.statistics.relational.cross_correlation import CrossCorrelationCalculator
from lumen.statistics.relational.granger_causality import GrangerCausalityCalculator
from lumen.statistics.relational.interaction_effects import InteractionEffectsCalculator
from lumen.statistics.relational.multicollinearity import MulticollinearityCalculator
from lumen.statistics.relational.mutual_information import MutualInformationCalculator
from lumen.statistics.relational.partial_correlation import PartialCorrelationCalculator


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
