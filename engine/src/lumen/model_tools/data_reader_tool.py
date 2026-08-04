# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Herramienta LangChain en Factory Maestra; depende de `ReadersInyeccionDependency`, `AnalyzeDataInyeccionDependency` y metadata factories.
# - ABSTRACCIÓN DEL DATO: Parámetros `backend` / `data` como handles abstractos (`rid` del `ObjectRegistry` o frame del backend), no pandas hardcodeado.
# - REFACTOR NATIVO: Tests de integración en los tres backends; sustituir `Generic` por ABC por backend en `create_data_context`.
# #[AI_CONTEXT_END]
from lumen.agents.master_factory import AgentMasterFactory
from langchain_core.tools import tool
from lumen.model_tools.object_registry import store

@tool
def data_reader_tool(file: str, backend: str) -> str:
    """ Ussage : data_reader_tool("file path", "backend implementation (polars, spark, pandas)")
        Returns an ID to use in other tools
    """

    # Validate backend
    if backend not in ["polars", "spark", "pandas"]:
        raise ValueError("Backend must be 'polars', 'spark', or 'pandas'.")

    try:
        master = AgentMasterFactory(backend)
        df = master.readers().read(file)
        return store(df)   
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")