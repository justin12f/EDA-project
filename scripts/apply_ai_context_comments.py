"""One-off helper: rewrite #[AI]/TODO/REFACTOR markers to AI_CONTEXT blocks. Not part of runtime."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD_MARKER = re.compile(
    r"^[ \t]*#.*(?:TODO\[AI\]|REFACTOR\[AI\]|FIX\[AI\]|FIC\[AI\]|# ?TODO\[AI\]|# ?REFACTOR\[AI\]|# ?FIX\[AI\]).*$",
    re.MULTILINE | re.IGNORECASE,
)
OLD_PLAIN_TODO = re.compile(
    r"^[ \t]*#TODO:.*$",
    re.MULTILINE,
)
AI_CONTEXT = re.compile(r"# #\[AI_CONTEXT_START\]")

DOMAIN_LABELS = {
    "descriptive": "DescriptiveStatistics",
    "inferential": "InferentialStatistics",
    "time_series": "TimeSeriesStatistics",
    "survival": "SurvivalStatistics",
    "segmentation": "SegmentationStatistics",
    "relational": "RelationalStatistics",
    "nlp": "NlpStatistics",
    "ml_support": "MlSupportStatistics",
    "graphs": "GraphStatistics",
    "geospatial": "GeospatialStatistics",
    "business": "BusinessStatistics",
}


def domain_from_path(path: Path) -> str:
    parts = path.parts
    if "statistics" in parts:
        idx = parts.index("statistics")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "Statistics"


def factory_name(path: Path) -> str:
    domain = domain_from_path(path)
    prefix = DOMAIN_LABELS.get(domain, "Statistics")
    return f"{prefix}Factory"


def block_for(path: Path, *, factory: str, data: str, native: str) -> str:
    return (
        "# #[AI_CONTEXT_START]\n"
        f"# - CONFIGURACIÓN DE FACTORY: {factory}\n"
        f"# - ABSTRACCIÓN DEL DATO: {data}\n"
        f"# - REFACTOR NATIVO: {native}\n"
        "# #[AI_CONTEXT_END]\n"
    )


def statistics_block(path: Path) -> str:
    fn = factory_name(path)
    return block_for(
        path,
        factory=(
            f"Registrar este calculator en `{fn}` (backends pandas | polars | spark) "
            f"y exponerlo mediante `StatisticsInyeccionDependency`, inyectada por la "
            f"Factory Maestra de Agentes junto a las demás fábricas de dominio."
        ),
        data=(
            "Mutar constructores y `analyze`/`compute` para recibir el contenedor abstracto "
            "del backend (`pd.DataFrame`, `pl.DataFrame`/`pl.LazyFrame`, `pyspark.sql.DataFrame`) "
            "inyectado por la factory; eliminar `np.ndarray`/`pd.Series` sueltos en firmas públicas."
        ),
        native=(
            "Resolver métricas con expresiones 100 % nativas del backend activo "
            "(Polars: `.select`/`.group_by`/`.agg` sin `.collect()` salvo materialización acordada; "
            "PySpark: `pyspark.sql.functions` y ventanas distribuidas; Pandas: operaciones vectorizadas). "
            "No convertir a NumPy/Pandas desde backends no-pandas."
        ),
    )


def path_specific_block(path: Path, rel: str) -> str | None:
    """Return a tailored AI_CONTEXT block for known modules."""
    r = rel.replace("\\", "/")

    if r == "algorithms/optimizers/gradient_descent.py":
        return block_for(
            path,
            factory=(
                "`OptimizerFactory` (pandas | polars | spark) bajo `ModelsInyeccionDependency`, "
                "encadenada desde la Factory Maestra de Agentes."
            ),
            data=(
                "Sustituir `np.ndarray`/`pd.Series` en `loss_function` y `GradientDescent` por tensores "
                "o columnas del backend inyectadas (p. ej. `pl.Expr`, columnas Spark, `pd.DataFrame`)."
            ),
            native=(
                "Implementar `GradientDescentPandas`, `GradientDescentPolars`, `GradientDescentSpark` "
                "con actualización de coeficientes vía operaciones nativas; la clase actual queda como "
                "contrato abstracto sin lógica NumPy en el camino de producción."
            ),
        )

    if r == "models/linear_regression.py":
        return block_for(
            path,
            factory=(
                "`LinearRegressionFactory` por (tipo, complejidad, backend) inyectada vía "
                "`LinearRegressionInyeccionDependency` desde la Factory Maestra de Agentes."
            ),
            data=(
                "Firmas `fit`/`predict` deben aceptar el contenedor del backend activo, no "
                "`np.ndarray`/`pd.DataFrame` genéricos mezclados."
            ),
            native=(
                "Tres implementaciones backend (`*Pandas`, `*Polars`, `*Spark`) que deleguen optimización "
                "y scoring a factories del mismo backend; sin importaciones directas cruzadas entre backends."
            ),
        )

    if r == "evaluation/score.py":
        return block_for(
            path,
            factory=(
                "`EvaluationScoreFactory` (pandas | polars | spark) dentro de "
                "`EvaluationInyeccionDependency`, inyectada por la Factory Maestra."
            ),
            data=(
                "Métricas deben operar sobre columnas o vectores materializados del backend, "
                "no solo `np.ndarray` en la API pública."
            ),
            native=(
                "MSE/MAE/R² con expresiones nativas (Polars `.pow`/`.mean`, Spark aggregations, "
                "Pandas vectorizado) según backend seleccionado por el agente."
            ),
        )

    if r.startswith("readers/"):
        return block_for(
            path,
            factory=(
                "`ReaderFactory` local por extensión y backend; `ReadersInyeccionDependency` "
                "como capa superior para Agentes."
            ),
            data=(
                "Retorno de `read()` debe ser `pl.LazyFrame` (polars), `pyspark.sql.DataFrame` (spark) "
                "o `pd.DataFrame` (pandas); sin tipos híbridos."
            ),
            native=(
                "Nuevos formatos como readers dedicados registrados en la factory del backend; "
                "lectura con APIs nativas (`pl.scan_*`, `spark.read`, `pd.read_*`)."
            ),
        )

    if r == "readers/reader_factory.py" and "pandas readers" in open(path, encoding="utf-8").read():
        pass  # handled by marker removal + single block at top

    if "data_cleaning/data_cleaning_step_factory" in r:
        return block_for(
            path,
            factory=(
                "Crear `AbstractDataCleaningStepFactory`, `PolarsDataCleaningStepFactory` y "
                "`SparkDataCleaningStepFactory` (más pandas); encadenar con "
                "`DataCleaningInyeccionDependency` → Factory Maestra."
            ),
            data=(
                "`create(step_name, data_frame, **kwargs)` debe tipar `data_frame` con el contenedor "
                "del backend, no `pd.DataFrame` fijo."
            ),
            native=(
                "Registros apuntan a steps en `data_cleaning/steps/backends/*`; eliminar "
                "`import pandas` de la factory abstracta."
            ),
        )

    if "data_cleaning/data_cleaning_pipeline" in r:
        return block_for(
            path,
            factory=(
                "`AbstractDataCleaningPipeline` + `PolarsDataCleaningPipeline` + "
                "`SparkDataCleaningPipeline` (+ pandas), inyectados por "
                "`DataCleaningInyeccionDependency`."
            ),
            data="Pipeline opera sobre el frame lazy/eager del backend inyectado en construcción.",
            native="Orquestación de steps solo vía factories del mismo backend; sin conversiones implícitas a pandas.",
        )

    if "data_cleaning/" in r and "steps" in r:
        backend = "abstract"
        if "backends/polars" in r:
            backend = "polars"
        elif "backends/spark" in r:
            backend = "spark"
        elif "backends/pandas" in r or r.endswith("steps/implementations.py"):
            backend = "pandas"
        return block_for(
            path,
            factory=(
                f"`DataCleaningStepFactory` del backend `{backend}` (registro en capa "
                f"`data_cleaning/steps/backends/`); inyección vía `DataCleaningInyeccionDependency`."
            ),
            data=(
                "Canonicalizar en `data_cleaning/steps/backends/`; deprecar duplicados en "
                "`steps/implementations.py` y `steps/polars_impl.py` raíz tras verificar referencias."
            ),
            native=(
                f"Steps en inglés y 100 % API nativa del backend `{backend}`; sin NumPy salvo "
                f"contrato explícito de materialización local."
            ),
        )

    if r.startswith("analyze_data/"):
        backend = "abstract"
        if "polars_impl" in r:
            backend = "polars"
        elif "spark_impl" in r:
            backend = "spark"
        elif "pandas_impl" in r or "implementations.py" in r:
            backend = "pandas"
        return block_for(
            path,
            factory=(
                f"`DataAnalyzerFactory` / `AnalyzeContextFactory` para backend `{backend}`; "
                f"inyectar vía `AnalyzeDataInyeccionDependency` desde Agentes."
            ),
            data=(
                "Analyzers reciben `pl.DataFrame`/`LazyFrame`, `Spark DataFrame` o `pd.DataFrame` "
                "según backend; eliminar conversiones `.to_pandas()` en ruta Polars/Spark."
            ),
            native=(
                f"Análisis con calculadoras de `statistics/` resueltas por backend; código y "
                f"docstrings en inglés; tests por implementación `{backend}`."
            ),
        )

    if r.startswith("model_tools/"):
        return block_for(
            path,
            factory=(
                "Herramienta LangChain registrada en la Factory Maestra de Agentes; depende de "
                "`ReadersInyeccionDependency`, `AnalyzeDataInyeccionDependency` y metadata factories."
            ),
            data="Parámetros `backend` / `data` deben ser handles abstractos, no frames pandas hardcodeados.",
            native="Validar con tests de integración los tres backends; corregir contrato tool↔factory si falla.",
        )

    if r == "agents/context_creator.py":
        return block_for(
            path,
            factory=(
                "`ContextCreatorAgent` consume `model_tools/*` inyectados por la Factory Maestra; "
                "orden estricto meta_data → data_reader → create_context."
            ),
            data="El agente debe propagar `backend` elegido a cada tool sin mezclar implementaciones.",
            native=(
                "Alinear tool-calling con factories reales; documentación y prompts solo en inglés; "
                "tests e2e del flujo de tres herramientas."
            ),
        )

    if "preproccesing/encoders" in r:
        return block_for(
            path,
            factory="`EncoderFactory` + `EncoderInyeccionDependency` (polars | spark) desde Factory Maestra.",
            data="Encoders reciben frames del backend registrado en `(encoder_type, backend)`.",
            native="Verificar `OneHotEncoderPolars`, `StandardScalerPolars`, etc.; corregir registro si falla create().",
        )

    if "prepare_input_for_context_model" in r:
        return block_for(
            path,
            factory="`ModelPreInputFactory` / contenedor DI al final del módulo, inyectado por Factory Maestra.",
            data="Schema de entrada robusto y tipado por backend, no dicts ad hoc.",
            native="Tests del pipeline de contexto ML; validar contra los tres backends configurables.",
        )

    if r == "parsers/parser.py":
        return block_for(
            path,
            factory=(
                "CLI debe resolver backend vía Factory Maestra (`--backend polars|spark|pandas`) "
                "y delegar a factories de pipeline/reader correspondientes."
            ),
            data="Argumentos de parser propagan backend al reader y cleaning factory inyectados.",
            native="Tests CLI con fixtures por backend; sin asumir pandas por defecto en código nuevo.",
        )

    if r == "gbm_config.py":
        return block_for(
            path,
            factory="Config global consumida por Factory Maestra; sin acoplamiento a un solo backend.",
            data="Reemplazar constantes NumPy por límites configurables por backend si aplica.",
            native="Mantener config declarativa; backends leen umbrales vía inyección.",
        )

    if r.startswith("statistics/"):
        return statistics_block(path)

    if "import numpy" in path.read_text(encoding="utf-8") or "import pandas" in path.read_text(
        encoding="utf-8"
    ):
        return statistics_block(path)

    return None


def strip_old_markers(text: str) -> str:
    text = OLD_MARKER.sub("", text)
    text = OLD_PLAIN_TODO.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def insert_block(text: str, block: str) -> str:
    if AI_CONTEXT.search(text):
        return text
    m = re.match(r'(\s*""".*?"""\s*\n)', text, re.DOTALL)
    if m:
        return text[: m.end()] + "\n" + block + text[m.end() :]
    return block + text


def process_file(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    if rel.startswith("tests/") or rel.startswith("scripts/"):
        return False
    if path.name == "generate_analyzers_backends.py":
        return False

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    has_np = "import numpy" in text or "from numpy" in text
    has_pd = "import pandas" in text or "from pandas" in text
    has_old = bool(OLD_MARKER.search(text) or OLD_PLAIN_TODO.search(text))

    if not (has_np or has_pd or has_old or "statistics/" in rel.replace("\\", "/")):
        return False

    block = path_specific_block(path, rel)
    if block is None and not has_old:
        return False

    new_text = strip_old_markers(text)
    if block:
        new_text = insert_block(new_text, block)
    new_text = strip_old_markers(new_text)  # remove any marker inside inserted paths

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed: list[str] = []
    for path in ROOT.rglob("*.py"):
        if process_file(path):
            changed.append(str(path.relative_to(ROOT)))
    for txt in [ROOT / "statistics" / "AI_READ.TXT", ROOT / "data_cleaning" / "steps" / "backends" / "AI_READ.TXT"]:
        if not txt.exists():
            continue
        body = block_for(
            txt,
            factory="Ver módulos del dominio: cada subcarpeta de `statistics/` o `data_cleaning/steps/backends/` tiene su Factory local.",
            data="Eliminar duplicados de implementación; una ruta canónica por backend.",
            native="Factories encadenadas → InyeccionDependency de dominio → Factory Maestra de Agentes.",
        )
        txt.write_text(body, encoding="utf-8")
        changed.append(str(txt.relative_to(ROOT)))

    print(f"Updated {len(changed)} files")
    for c in sorted(changed):
        print(f"  {c}")


if __name__ == "__main__":
    main()
