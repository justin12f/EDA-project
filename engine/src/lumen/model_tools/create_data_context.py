# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Herramienta LangChain en Factory Maestra; depende de `ReadersInyeccionDependency`, `AnalyzeDataInyeccionDependency` y metadata factories.
# - ABSTRACCIÓN DEL DATO: Parámetros `backend` / `data` como handles abstractos (`rid` del `ObjectRegistry` o frame del backend), no pandas hardcodeado.
# - REFACTOR NATIVO: Tests de integración en los tres backends; sustituir `Generic` por ABC por backend en `create_data_context`.
# #[AI_CONTEXT_END]
"module for create superficial context of the data for my models"
from abc import ABC , abstractmethod
from langchain_core.tools import tool
from lumen.model_tools.object_registry import retrieve
# import pandas implementations for create context
from lumen.analyze_data.analyzers.backends.pandas_impl import (
    AnalyseDataColumns,
    AnalyseDataDescribe,
    AnalyseDataHead,
    AnalyseDataIndex,
    AnalyseDataInfo,
    AnalyseDataSample,
    AnalyseDataShape,
    AnalyseDataTail,
    AnalyseDataTypes
    )

# import polars implementations for create context
from lumen.analyze_data.analyzers.backends.polars_impl import (
    AnalyseDataColumnsPolars,
    AnalyseDataDescribePolars,
    AnalyseDataHeadPolars,
    AnalyseDataIndexPolars,
    AnalyseDataInfoPolars,
    AnalyseDataSamplePolars,
    AnalyseDataShapePolars,
    AnalyseDataTailPolars,
    AnalyseDataTypesPolars
    )
from typing import Generic

class AnalyzeContextAbstract:
    """Base class for all AnalyzeContext classes"""
    @abstractmethod
    def analyze_context(self,Generic):
        """
        Analyze the data and return the context.
        
        """
        
    

class AnalyzeContextPandas(AnalyzeContextAbstract):
    def analyze_context(self, Generic):
        context: dict[str, str] = {}

        analyzer = AnalyseDataColumns()
        analyzer.set_data_frame(Generic)
        context["columns"] = analyzer.analyze()

        analyzer = AnalyseDataDescribe()
        analyzer.set_data_frame(Generic)
        context["describe"] = analyzer.analyze()

        analyzer = AnalyseDataHead()
        analyzer.set_data_frame(Generic)
        context["head"] = analyzer.analyze()

        analyzer = AnalyseDataIndex()
        analyzer.set_data_frame(Generic)
        context["index"] = analyzer.analyze()

        analyzer = AnalyseDataInfo()
        analyzer.set_data_frame(Generic)
        context["info"] = analyzer.analyze()

        analyzer = AnalyseDataSample()
        analyzer.set_data_frame(Generic)
        context["sample"] = analyzer.analyze()

        analyzer = AnalyseDataShape()
        analyzer.set_data_frame(Generic)
        context["shape"] = analyzer.analyze()

        analyzer = AnalyseDataTail()
        analyzer.set_data_frame(Generic)
        context["tail"] = analyzer.analyze()

        analyzer = AnalyseDataTypes()
        analyzer.set_data_frame(Generic)
        context["types"] = analyzer.analyze()

        return context
        
class AnalyzeContextSpark(AnalyzeContextAbstract):
    def analyze_context(self, Generic):
        # PySpark is an optional dependency (the ``spark`` extra) — imported
        # lazily here so importing this module never requires it.
        try:
            from lumen.analyze_data.analyzers.backends.spark_impl import (
                AnalyseDataColumnsSpark,
                AnalyseDataDescribeSpark,
                AnalyseDataHeadSpark,
                AnalyseDataIndexSpark,
                AnalyseDataInfoSpark,
                AnalyseDataSampleSpark,
                AnalyseDataShapeSpark,
                AnalyseDataTailSpark,
                AnalyseDataTypesSpark,
            )
        except ImportError as exc:
            raise ImportError(
                "Backend 'spark' requires PySpark. "
                "Install it with: uv sync --extra spark"
            ) from exc

        context: dict[str, str] = {}

        analyzer = AnalyseDataColumnsSpark()
        analyzer.set_data_frame(Generic)
        context["columns"] = analyzer.analyze()

        analyzer = AnalyseDataDescribeSpark()
        analyzer.set_data_frame(Generic)
        context["describe"] = analyzer.analyze()

        analyzer = AnalyseDataHeadSpark()
        analyzer.set_data_frame(Generic)
        context["head"] = analyzer.analyze()

        analyzer = AnalyseDataIndexSpark()
        analyzer.set_data_frame(Generic)
        context["index"] = analyzer.analyze()

        analyzer = AnalyseDataInfoSpark()
        analyzer.set_data_frame(Generic)
        context["info"] = analyzer.analyze()

        analyzer = AnalyseDataSampleSpark()
        analyzer.set_data_frame(Generic)
        context["sample"] = analyzer.analyze()

        analyzer = AnalyseDataShapeSpark()
        analyzer.set_data_frame(Generic)
        context["shape"] = analyzer.analyze()

        analyzer = AnalyseDataTailSpark()
        analyzer.set_data_frame(Generic)
        context["tail"] = analyzer.analyze()

        analyzer = AnalyseDataTypesSpark()
        analyzer.set_data_frame(Generic)
        context["types"] = analyzer.analyze()

        return context

class AnalyzeContextPolars(AnalyzeContextAbstract):
    def analyze_context(self, Generic):

        context: dict[str, str] = {}

        # columns
        analyzer = AnalyseDataColumnsPolars()
        analyzer.set_data_frame(Generic)
        context["columns"] = analyzer.analyze()

        # describe
        analyzer = AnalyseDataDescribePolars()
        analyzer.set_data_frame(Generic)
        context["describe"] = analyzer.analyze()

        # head
        analyzer = AnalyseDataHeadPolars()
        analyzer.set_data_frame(Generic)
        context["head"] = analyzer.analyze()

        # index
        analyzer = AnalyseDataIndexPolars()
        analyzer.set_data_frame(Generic)
        context["index"] = analyzer.analyze()

        # info
        analyzer = AnalyseDataInfoPolars()
        analyzer.set_data_frame(Generic)
        context["info"] = analyzer.analyze()

        # sample
        analyzer = AnalyseDataSamplePolars()
        analyzer.set_data_frame(Generic)
        context["sample"] = analyzer.analyze()

        # shape
        analyzer = AnalyseDataShapePolars()
        analyzer.set_data_frame(Generic)
        context["shape"] = analyzer.analyze()

        # tail
        analyzer = AnalyseDataTailPolars()
        analyzer.set_data_frame(Generic)
        context["tail"] = analyzer.analyze()

        # types
        analyzer = AnalyseDataTypesPolars()
        analyzer.set_data_frame(Generic)
        context["types"] = analyzer.analyze()

        return context

#Inyeccion dependency for backends

class CreateContext:
    """Class to create context for the models

    Args:
        Generic: The data to analyze
        backend_implementation: The backend to use for analysis

        Ussage Example:

                context = CreateContext().analyze_context(data,"the backend implementation selected it  AnalyzeContextPandas ,AnalyzeContextSpark,AnalyzeContextPolars ")
        
        """

    def __init__(self,Generic,backend_implementation:AnalyzeContextAbstract):
        self.Generic = Generic
        self.context = ""
        self.backend_implementation:AnalyzeContextAbstract = backend_implementation

    def analyze(self):
        self.context = self.backend_implementation.analyze_context(self.Generic)

        return self.context

#Tool expose for de model

@tool
def create_context_agent_tool(data: str, backend_implementation: str) -> dict:
    """
    Args:
        data: identificadtor of dataFrame object
        backend_implementation: name of the class (ej. 'AnalyzeContextPolars')
    """
    df = retrieve(data) 
    implementation_map = {
        "AnalyzeContextPolars": AnalyzeContextPolars(),
        "AnalyzeContextSpark": AnalyzeContextSpark(),
        "AnalyzeContextPandas": AnalyzeContextPandas(),
    }
    impl = implementation_map[backend_implementation]
    return CreateContext(df, impl).analyze()