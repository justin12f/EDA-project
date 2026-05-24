$Root = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding $false

$replacements = @{
    'CONFIGURACIÃ"N' = 'CONFIGURACIÓN'
    'CONFIGURACIÃƒâ€šÃ‚Â"N' = 'CONFIGURACIÓN'
    'ABSTRACCIÃ"N' = 'ABSTRACCIÓN'
    'ABSTRACCIÃƒâ€šÃ‚Â"N' = 'ABSTRACCIÓN'
    'vÃ­a' = 'vía'
    'vÃƒÂ­a' = 'vía'
    'segÃºn' = 'según'
    'genÃ©ricos' = 'genéricos'
    'optimizaciÃ³n' = 'optimización'
    'documentaciÃ³n' = 'documentación'
    'inglÃ©s' = 'inglés'
    'implementaciÃ³n' = 'implementación'
    'AnÃ¡lisis' = 'Análisis'
    'cÃ³digo' = 'código'
    'â†'' = '→'
    'Ã³' = 'ó'
    'Ã­' = 'í'
    'Ã©' = 'é'
    'Ã¡' = 'á'
    'Ãº' = 'ú'
    'Ã±' = 'ñ'
}

$gradientBlock = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `OptimizerFactory` (pandas | polars | spark) bajo `ModelsInyeccionDependency`, encadenada desde la Factory Maestra de Agentes.
# - ABSTRACCIÓN DEL DATO: Sustituir `np.ndarray`/`pd.Series` en `loss_function` y `GradientDescent` por tensores o columnas del backend inyectadas (p. ej. `pl.Expr`, columnas Spark, columnas `pd.DataFrame`).
# - REFACTOR NATIVO: Implementar `GradientDescentPandas`, `GradientDescentPolars`, `GradientDescentSpark` con actualización de coeficientes vía operaciones nativas; la clase actual queda como contrato abstracto sin lógica NumPy en el camino de producción.
# #[AI_CONTEXT_END]

"@

$registryBlock = @"
# #[AI_CONTEXT_START]
# - CONFIGURACIÓN DE FACTORY: `ObjectRegistry` como servicio transversal inyectado en `model_tools` y la Factory Maestra de Agentes para pasar handles entre tools.
# - ABSTRACCIÓN DEL DATO: Almacenar referencias al contenedor del backend activo (`rid` → frame abstracto), no serializar `pd.DataFrame` por defecto.
# - REFACTOR NATIVO: Optimizar almacenamiento/recuperación; documentación y mensajes de error en inglés.
# #[AI_CONTEXT_END]

"@

$changed = 0
Get-ChildItem -Path $Root -Recurse -Include *.py,*.TXT -File | ForEach-Object {
    if ($_.FullName -match '\\scripts\\') { return }
    $text = [IO.File]::ReadAllText($_.FullName, [Text.Encoding]::UTF8)
    $orig = $text
    foreach ($k in $replacements.Keys) {
        $text = $text.Replace($k, $replacements[$k])
    }
    if ($text -match 'REFACTOR\[AI\]|TODO\[AI\]|FIX\[AI\]') {
        $text = [regex]::Replace($text, '(?m)^[ \t]*#.*(?:TODO\[AI\]|REFACTOR\[AI\]|FIX\[AI\]).*\r?\n', '')
    }
    if ($text -match '(?m)^[ \t]*#TODO:') {
        $text = [regex]::Replace($text, '(?m)^[ \t]*#TODO:.*\r?\n', '')
    }
    $rel = $_.FullName.Substring($Root.Length + 1).Replace('\', '/')
    if ($rel -eq 'algorithms/optimizers/gradient_descent.py' -and $text -notmatch 'AI_CONTEXT_START') {
        $text = $text -replace '"""Module to create the gradient descent algorithm"""\r?\n', "`"``"Module to create the gradient descent algorithm`"`"`n$gradientBlock"
    }
    if ($rel -eq 'model_tools/object_registry.py' -and $text -notmatch 'AI_CONTEXT_START') {
        $text = "# model_tools/object_registry.py`n$registryBlock" + ($text -replace '^# model_tools/object_registry\.py\r?\n', '')
    }
    if ($text -ne $orig) {
        [IO.File]::WriteAllText($_.FullName, $text, $utf8)
        $changed++
    }
}
Write-Host "Fixed $changed files"
