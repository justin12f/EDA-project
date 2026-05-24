"""Master factory: chains domain *InyeccionDependency layers for AI agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.backend import Backend, DEFAULT_BACKEND, validate_backend

if TYPE_CHECKING:
    from data_cleaning.inyeccion import DataCleaningInyeccionDependency
    from analyze_data.inyeccion import AnalyzeDataInyeccionDependency
    from evaluation.inyeccion import EvaluationInyeccionDependency
    from models.inyeccion import ModelsInyeccionDependency
    from preproccesing.encoders.inyeccion import EncoderInyeccionDependency
    from readers.inyeccion import ReadersInyeccionDependency
    from statistics.inyeccion import StatisticsInyeccionDependency


class AgentMasterFactory:
    """Composition root for agent tools and pipelines."""

    def __init__(self, backend: Backend | str = DEFAULT_BACKEND) -> None:
        self._backend: Backend = validate_backend(str(backend))
        self._readers: ReadersInyeccionDependency | None = None
        self._cleaning: DataCleaningInyeccionDependency | None = None
        self._analyzers: AnalyzeDataInyeccionDependency | None = None
        self._statistics: StatisticsInyeccionDependency | None = None
        self._encoders: EncoderInyeccionDependency | None = None
        self._models: ModelsInyeccionDependency | None = None
        self._evaluation: EvaluationInyeccionDependency | None = None

    @property
    def backend(self) -> Backend:
        return self._backend

    def readers(self) -> ReadersInyeccionDependency:
        if self._readers is None:
            from readers.inyeccion import ReadersInyeccionDependency

            self._readers = ReadersInyeccionDependency(self._backend)
        return self._readers

    def cleaning(self) -> DataCleaningInyeccionDependency:
        if self._cleaning is None:
            from data_cleaning.inyeccion import DataCleaningInyeccionDependency

            self._cleaning = DataCleaningInyeccionDependency(self._backend)
        return self._cleaning

    def analyzers(self) -> AnalyzeDataInyeccionDependency:
        if self._analyzers is None:
            from analyze_data.inyeccion import AnalyzeDataInyeccionDependency

            self._analyzers = AnalyzeDataInyeccionDependency(self._backend)
        return self._analyzers

    def statistics(self) -> StatisticsInyeccionDependency:
        if self._statistics is None:
            from statistics.inyeccion import StatisticsInyeccionDependency

            self._statistics = StatisticsInyeccionDependency(self._backend)
        return self._statistics

    def encoders(self) -> EncoderInyeccionDependency:
        if self._encoders is None:
            from preproccesing.encoders.inyeccion import EncoderInyeccionDependency

            self._encoders = EncoderInyeccionDependency(self._backend)
        return self._encoders

    def models(self) -> ModelsInyeccionDependency:
        if self._models is None:
            from models.inyeccion import ModelsInyeccionDependency

            self._models = ModelsInyeccionDependency(self._backend)
        return self._models

    def evaluation(self) -> EvaluationInyeccionDependency:
        if self._evaluation is None:
            from evaluation.inyeccion import EvaluationInyeccionDependency

            self._evaluation = EvaluationInyeccionDependency(self._backend)
        return self._evaluation
