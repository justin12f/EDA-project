# model_tools/object_registry.py

# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `ObjectRegistry` como servicio transversal inyectado en `model_tools` y la Factory Maestra de Agentes para pasar handles entre tools.
# - ABSTRACCIÓN DEL DATO: Almacenar referencias al contenedor del backend activo (`rid` → frame abstracto), no serializar `pd.DataFrame` por defecto.
# - REFACTOR NATIVO: Optimizar almacenamiento/recuperación; documentación y mensajes de error en inglés.
# #[AI_CONTEXT_END]
_data_store: dict[str, any] = {}


def store(obj: any) -> str:
    """Store an object and return a unique identifier."""
    import uuid
    rid = uuid.uuid4().hex[:8]
    _data_store[rid] = obj
    return rid

def retrieve(rid: str) -> any:
    """Retrieve an object by its identifier."""
    if rid not in _data_store:
        raise KeyError(f"Data ID '{rid}' not found.")
    return _data_store[rid]