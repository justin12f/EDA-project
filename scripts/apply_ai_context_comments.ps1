$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $Root "statistics"))) {
    $Root = Split-Path -Parent $PSScriptRoot
}

$domainLabels = @{
    descriptive = "DescriptiveStatistics"
    inferential = "InferentialStatistics"
    time_series = "TimeSeriesStatistics"
    survival = "SurvivalStatistics"
    segmentation = "SegmentationStatistics"
    relational = "RelationalStatistics"
    nlp = "NlpStatistics"
    ml_support = "MlSupportStatistics"
    graphs = "GraphStatistics"
    geospatial = "GeospatialStatistics"
    business = "BusinessStatistics"
}

function Get-StatisticsBlock($relPath) {
    $domain = "Statistics"
  if ($relPath -match "statistics[\\/]([^\\/]+)[\\/]") {
        $d = $Matches[1]
        if ($domainLabels.ContainsKey($d)) { $domain = $domainLabels[$d] }
    }
    $fn = "${domain}Factory"
    return @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Registrar este calculator en ``$fn`` (backends pandas | polars | spark) y exponerlo mediante ``StatisticsInyeccionDependency``, inyectada por la Factory Maestra de Agentes junto a las demás fábricas de dominio.
# - ABSTRACCIÓN DEL DATO: Mutar constructores y ``analyze``/``compute`` para recibir el contenedor abstracto del backend (``pd.DataFrame``, ``pl.DataFrame``/``pl.LazyFrame``, ``pyspark.sql.DataFrame``) inyectado por la factory; eliminar ``np.ndarray``/``pd.Series`` sueltos en firmas públicas.
# - REFACTOR NATIVO: Resolver métricas con expresiones 100 % nativas del backend activo (Polars: ``.select``/``.group_by``/``.agg`` sin ``.collect()`` salvo materialización acordada; PySpark: ``pyspark.sql.functions`` y ventanas distribuidas; Pandas: operaciones vectorizadas). No convertir a NumPy/Pandas desde backends no-pandas.
# #[AI_CONTEXT_END]

"@
}

$markerPattern = '(?m)^[ \t]*#.*(?:TODO\[AI\]|REFACTOR\[AI\]|FIX\[AI\]|FIC\[AI\]).*\r?\n'

$changed = @()

Get-ChildItem -Path $Root -Recurse -Filter *.py | ForEach-Object {
    $rel = $_.FullName.Substring($Root.Length + 1).Replace('\', '/')
    if ($rel -match '^(tests/|scripts/)') { return }
    if ($_.Name -eq 'generate_analyzers_backends.py') { return }

    $text = [IO.File]::ReadAllText($_.FullName)
    $hasNp = $text -match 'import numpy|from numpy'
    $hasPd = $text -match 'import pandas|from pandas'
    $hasOld = $text -match 'TODO\[AI\]|REFACTOR\[AI\]|FIX\[AI\]|FIC\[AI\]'
    $isStats = $rel -like 'statistics/*'

    if (-not ($hasNp -or $hasPd -or $hasOld -or $isStats)) { return }

    $newText = [regex]::Replace($text, $markerPattern, '')
    $newText = [regex]::Replace($newText, '(?m)^[ \t]*#TODO:.*\r?\n', '')

    if ($newText -notmatch '# #\[AI_CONTEXT_START\]') {
        $block = $null
        switch -Regex ($rel) {
            'algorithms/optimizers/gradient_descent.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``OptimizerFactory`` (pandas | polars | spark) bajo ``ModelsInyeccionDependency``, encadenada desde la Factory Maestra de Agentes.
# - ABSTRACCIÓN DEL DATO: Sustituir ``np.ndarray``/``pd.Series`` en ``loss_function`` y ``GradientDescent`` por tensores o columnas del backend inyectadas (p. ej. ``pl.Expr``, columnas Spark, columnas ``pd.DataFrame``).
# - REFACTOR NATIVO: Implementar ``GradientDescentPandas``, ``GradientDescentPolars``, ``GradientDescentSpark`` con actualización de coeficientes vía operaciones nativas; la clase actual queda como contrato abstracto sin lógica NumPy en el camino de producción.
# #[AI_CONTEXT_END]

"@
            }
            'models/linear_regression.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``LinearRegressionFactory`` por (tipo, complejidad, backend) inyectada vía ``LinearRegressionInyeccionDependency`` desde la Factory Maestra de Agentes.
# - ABSTRACCIÓN DEL DATO: Firmas ``fit``/``predict`` deben aceptar el contenedor del backend activo, no ``np.ndarray``/``pd.DataFrame`` genéricos mezclados.
# - REFACTOR NATIVO: Tres implementaciones backend (``*Pandas``, ``*Polars``, ``*Spark``) que deleguen optimización y scoring a factories del mismo backend; sin importaciones directas cruzadas entre backends.
# #[AI_CONTEXT_END]

"@
            }
            'evaluation/score.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``EvaluationScoreFactory`` (pandas | polars | spark) dentro de ``EvaluationInyeccionDependency``, inyectada por la Factory Maestra.
# - ABSTRACCIÓN DEL DATO: Métricas deben operar sobre columnas o vectores materializados del backend, no solo ``np.ndarray`` en la API pública.
# - REFACTOR NATIVO: MSE/MAE/R² con expresiones nativas (Polars ``.pow``/``.mean``, Spark aggregations, Pandas vectorizado) según backend seleccionado por el agente.
# #[AI_CONTEXT_END]

"@
            }
            '^readers/' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``ReaderFactory`` local por extensión y backend; ``ReadersInyeccionDependency`` como capa superior para Agentes.
# - ABSTRACCIÓN DEL DATO: Retorno de ``read()`` debe ser ``pl.LazyFrame`` (polars), ``pyspark.sql.DataFrame`` (spark) o ``pd.DataFrame`` (pandas); sin tipos híbridos.
# - REFACTOR NATIVO: Nuevos formatos como readers dedicados registrados en la factory del backend; lectura con APIs nativas (``pl.scan_*``, ``spark.read``, ``pd.read_*``).
# #[AI_CONTEXT_END]

"@
            }
            'data_cleaning/data_cleaning_step_factory.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Crear ``AbstractDataCleaningStepFactory``, ``PolarsDataCleaningStepFactory`` y ``SparkDataCleaningStepFactory`` (más pandas); encadenar con ``DataCleaningInyeccionDependency`` → Factory Maestra.
# - ABSTRACCIÓN DEL DATO: ``create(step_name, data_frame, **kwargs)`` debe tipar ``data_frame`` con el contenedor del backend, no ``pd.DataFrame`` fijo.
# - REFACTOR NATIVO: Registros apuntan a steps en ``data_cleaning/steps/backends/*``; eliminar ``import pandas`` de la factory abstracta.
# #[AI_CONTEXT_END]

"@
            }
            'data_cleaning/data_cleaning_pipeline.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``AbstractDataCleaningPipeline`` + ``PolarsDataCleaningPipeline`` + ``SparkDataCleaningPipeline`` (+ pandas), inyectados por ``DataCleaningInyeccionDependency``.
# - ABSTRACCIÓN DEL DATO: Pipeline opera sobre el frame lazy/eager del backend inyectado en construcción.
# - REFACTOR NATIVO: Orquestación de steps solo vía factories del mismo backend; sin conversiones implícitas a pandas.
# #[AI_CONTEXT_END]

"@
            }
            'data_cleaning/data_cleaning_report.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``DataCleaningReportFactory`` por backend, inyectada vía ``DataCleaningInyeccionDependency`` desde la Factory Maestra.
# - ABSTRACCIÓN DEL DATO: Métricas before/after sobre el contenedor del backend, no ``DataFrame`` pandas fijo ni ``np.ndarray`` auxiliar.
# - REFACTOR NATIVO: Comparación de métricas con agregaciones nativas del backend activo.
# #[AI_CONTEXT_END]

"@
            }
            'data_cleaning/wrapper_steps_with_logger.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Wrapper registrado en ``DataCleaningStepFactory`` del backend activo; inyección vía ``DataCleaningInyeccionDependency``.
# - ABSTRACCIÓN DEL DATO: ``wrapped(data)`` debe aceptar el frame del backend inyectado, no ``pandas.DataFrame`` exclusivo.
# - REFACTOR NATIVO: Logging y ``compare_metrics`` sin copias pandas obligatorias; clonar/materializar según contrato del backend.
# #[AI_CONTEXT_END]

"@
            }
            'data_cleaning/' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``DataCleaningStepFactory`` del backend correspondiente en ``data_cleaning/steps/backends/``; inyección vía ``DataCleaningInyeccionDependency``.
# - ABSTRACCIÓN DEL DATO: Canonicalizar implementaciones en ``backends/``; deprecar duplicados en ``steps/implementations.py`` y ``steps/polars_impl.py`` raíz tras verificar referencias.
# - REFACTOR NATIVO: Steps en inglés y 100 % API nativa del backend; sin NumPy salvo materialización local explícita.
# #[AI_CONTEXT_END]

"@
            }
            'analyze_data/' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``DataAnalyzerFactory`` / ``AnalyzeContextFactory`` por backend; inyectar vía ``AnalyzeDataInyeccionDependency`` desde Agentes.
# - ABSTRACCIÓN DEL DATO: Analyzers reciben ``pl.DataFrame``/``LazyFrame``, Spark ``DataFrame`` o ``pd.DataFrame`` según backend; eliminar ``.to_pandas()`` en ruta Polars/Spark.
# - REFACTOR NATIVO: Análisis con calculadoras de ``statistics/`` resueltas por backend; código y docstrings en inglés; tests por implementación.
# #[AI_CONTEXT_END]

"@
            }
            'model_tools/' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Herramienta LangChain registrada en la Factory Maestra de Agentes; depende de ``ReadersInyeccionDependency``, ``AnalyzeDataInyeccionDependency`` y metadata factories.
# - ABSTRACCIÓN DEL DATO: Parámetros ``backend`` / ``data`` deben ser handles abstractos, no frames pandas hardcodeados.
# - REFACTOR NATIVO: Validar con tests de integración los tres backends; corregir contrato tool↔factory si falla.
# #[AI_CONTEXT_END]

"@
            }
            'agents/context_creator.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``ContextCreatorAgent`` consume ``model_tools/*`` inyectados por la Factory Maestra; orden estricto meta_data → data_reader → create_context.
# - ABSTRACCIÓN DEL DATO: El agente debe propagar ``backend`` elegido a cada tool sin mezclar implementaciones.
# - REFACTOR NATIVO: Alinear tool-calling con factories reales; documentación y prompts solo en inglés; tests e2e del flujo de tres herramientas.
# #[AI_CONTEXT_END]

"@
            }
            'preproccesing/' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: ``EncoderFactory`` / ``ModelPreInputFactory`` + capa ``*InyeccionDependency``, inyectadas por la Factory Maestra de Agentes.
# - ABSTRACCIÓN DEL DATO: Entrada/salida tipada por backend (``pl.DataFrame``, Spark ``DataFrame``, ``pd.DataFrame``); schema robusto en pre-input.
# - REFACTOR NATIVO: Verificar encoders y pipeline de contexto ML con tests por backend; corregir registro en factory si ``create()`` falla.
# #[AI_CONTEXT_END]

"@
            }
            'parsers/parser.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: CLI resuelve backend vía Factory Maestra (``--backend polars|spark|pandas``) y delega a factories de pipeline/reader.
# - ABSTRACCIÓN DEL DATO: Argumentos propagan backend al reader y cleaning factory inyectados.
# - REFACTOR NATIVO: Tests CLI con fixtures por backend; sin asumir pandas por defecto en código nuevo.
# #[AI_CONTEXT_END]

"@
            }
            'gbm_config.py' {
                $block = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Config global consumida por Factory Maestra; sin acoplamiento a un solo backend.
# - ABSTRACCIÓN DEL DATO: Reemplazar constantes NumPy por límites configurables por backend si aplica.
# - REFACTOR NATIVO: Mantener config declarativa; backends leen umbrales vía inyección.
# #[AI_CONTEXT_END]

"@
            }
            '^statistics/' {
                $block = Get-StatisticsBlock $rel
            }
            default {
                if ($hasNp -or $hasPd) { $block = Get-StatisticsBlock $rel }
            }
        }

        if ($block) {
            if ($newText -match '(?s)^(\s*""".*?"""\s*\r?\n)') {
                $newText = $Matches[1] + "`n" + $block + $newText.Substring($Matches[1].Length)
            } else {
                $newText = $block + $newText
            }
        }
    }

    $newText = [regex]::Replace($newText, $markerPattern, '')
    $newText = [regex]::Replace($newText, '(?m)^[ \t]*#TODO:.*\r?\n', '')
    $newText = [regex]::Replace($newText, '(\r?\n){3,}', "`n`n")

    if ($newText -ne $text) {
        [IO.File]::WriteAllText($_.FullName, $newText)
        $changed += $rel
    }
}

$aiTxt = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: Cada subcarpeta de ``statistics/`` o ``data_cleaning/steps/backends/`` define su Factory local (pandas | polars | spark).
# - ABSTRACCIÓN DEL DATO: Una implementación canónica por backend; eliminar duplicados de nombre/ruta.
# - REFACTOR NATIVO: Encadenar Factory local → ``*InyeccionDependency`` de dominio → Factory Maestra de Agentes.
# #[AI_CONTEXT_END]
"@

foreach ($p in @(
    (Join-Path $Root 'statistics\AI_READ.TXT'),
    (Join-Path $Root 'data_cleaning\steps\backends\AI_READ.TXT')
)) {
    if (Test-Path $p) {
        [IO.File]::WriteAllText($p, $aiTxt.TrimStart("`n"))
        $changed += (Split-Path $p -Leaf)
    }
}

Write-Host "Updated $($changed.Count) files"
