# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Herramienta LangChain en Factory Maestra; depende de `ReadersInyeccionDependency`, `AnalyzeDataInyeccionDependency` y metadata factories.
# - ABSTRACCIÓN DEL DATO: Parámetros `backend` / `data` como handles abstractos (`rid` del `ObjectRegistry` o frame del backend), no pandas hardcodeado.
# - REFACTOR NATIVO: Tests de integración en los tres backends; sustituir `Generic` por ABC por backend en `create_data_context`.
# #[AI_CONTEXT_END]
from lumen.preproccesing.model_pre_input.prepare_input_for_context_model import DataContextService
from langchain_core.tools import tool
from lumen.preproccesing.model_pre_input.prepare_input_for_context_model import MetadataToolInputSchema
from lumen.preproccesing.model_pre_input.prepare_input_for_context_model import DIContainer  # Asegúrate de importarlo

@tool(args_schema=MetadataToolInputSchema, return_direct=False)
def meta_data_context(path: str) -> str:
    data_context_service = DIContainer.build_data_context_service()  # ← Corrección aquí
    return data_context_service.generate_agent_prompt(
        uri_or_path=path,
        base_system_prompt="Please analyze the following source metadata:"
    )
