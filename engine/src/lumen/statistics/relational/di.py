"""Dependency injection container for the relational statistics domain."""
from __future__ import annotations

from typing import Literal

from relational.abstract.contingency_analysis import AbstractContingencyAnalysisCalculator
from relational.abstract.correlation_matrix import AbstractCorrelationMatrixCalculator
from relational.abstract.cross_correlation import AbstractCrossCorrelationCalculator
from relational.abstract.granger_causality import AbstractGrangerCausalityCalculator
from relational.abstract.interaction_effects import AbstractInteractionEffectsCalculator
from relational.abstract.multicollinearity import AbstractMulticollinearityCalculator
from relational.abstract.mutual_information import AbstractMutualInformationCalculator
from relational.abstract.partial_correlation import AbstractPartialCorrelationCalculator
from relational.factory import RelationalStatisticsFactory

Backend = Literal["polars", "spark", "pandas"]


class RelationalDependencyContainer:
    def __init__(self, backend: Backend) -> None:
        self._backend: Backend = backend
        self._factory = RelationalStatisticsFactory

        self._contingency_analysis: AbstractContingencyAnalysisCalculator | None = None
        self._correlation_matrix: AbstractCorrelationMatrixCalculator | None = None
        self._cross_correlation: AbstractCrossCorrelationCalculator | None = None
        self._granger_causality: AbstractGrangerCausalityCalculator | None = None
        self._interaction_effects: AbstractInteractionEffectsCalculator | None = None
        self._multicollinearity: AbstractMulticollinearityCalculator | None = None
        self._mutual_information: AbstractMutualInformationCalculator | None = None
        self._partial_correlation: AbstractPartialCorrelationCalculator | None = None

    @property
    def backend(self) -> Backend:
        return self._backend

    def contingency_analysis_calculator(self) -> AbstractContingencyAnalysisCalculator:
        if self._contingency_analysis is None:
            self._contingency_analysis = self._factory.create("contingency_analysis_calculator", self._backend)
        return self._contingency_analysis

    def correlation_matrix_calculator(self) -> AbstractCorrelationMatrixCalculator:
        if self._correlation_matrix is None:
            self._correlation_matrix = self._factory.create("correlation_matrix_calculator", self._backend)
        return self._correlation_matrix

    def cross_correlation_calculator(self) -> AbstractCrossCorrelationCalculator:
        if self._cross_correlation is None:
            self._cross_correlation = self._factory.create("cross_correlation_calculator", self._backend)
        return self._cross_correlation

    def granger_causality_calculator(self) -> AbstractGrangerCausalityCalculator:
        if self._granger_causality is None:
            self._granger_causality = self._factory.create("granger_causality_calculator", self._backend)
        return self._granger_causality

    def interaction_effects_calculator(self) -> AbstractInteractionEffectsCalculator:
        if self._interaction_effects is None:
            self._interaction_effects = self._factory.create("interaction_effects_calculator", self._backend)
        return self._interaction_effects

    def multicollinearity_calculator(self) -> AbstractMulticollinearityCalculator:
        if self._multicollinearity is None:
            self._multicollinearity = self._factory.create("multicollinearity_calculator", self._backend)
        return self._multicollinearity

    def mutual_information_calculator(self) -> AbstractMutualInformationCalculator:
        if self._mutual_information is None:
            self._mutual_information = self._factory.create("mutual_information_calculator", self._backend)
        return self._mutual_information

    def partial_correlation_calculator(self) -> AbstractPartialCorrelationCalculator:
        if self._partial_correlation is None:
            self._partial_correlation = self._factory.create("partial_correlation_calculator", self._backend)
        return self._partial_correlation
