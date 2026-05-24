$Root = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding $false
$arrow = [char]0x2192

$map = [ordered]@{
    'CONFIGURACIÃ"N DE FACTORY' = 'CONFIGURACIÓN DE FACTORY'
    'ABSTRACCIÃ"N DEL DATO' = 'ABSTRACCIÓN DEL DATO'
    'demÃ¡s fÃ¡bricas' = 'demás fábricas'
    'firmas pÃºblicas' = 'firmas públicas'
    'Resolver mÃ©tricas' = 'Resolver métricas'
    'materializaciÃ³n acordada' = 'materialización acordada'
    'inyectar vÃ­a' = 'inyectar vía'
    'segÃºn backend' = 'según backend'
    'genÃ©ricos mezclados' = 'genéricos mezclados'
    'optimizaciÃ³n y scoring' = 'optimización y scoring'
    'documentaciÃ³n y prompts' = 'documentación y prompts'
    'inglÃ©s' = 'inglés'
    'AnÃ¡lisis con' = 'Análisis con'
    'cÃ³digo y docstrings' = 'código y docstrings'
    'implementaciÃ³n' = 'implementación'
}

$n = 0
Get-ChildItem -Path $Root -Recurse -Include *.py,*.TXT | ForEach-Object {
    if ($_.FullName -match '\\scripts\\') { return }
    $t = [IO.File]::ReadAllText($_.FullName, $utf8)
    if ($t -notmatch 'AI_CONTEXT_START') { return }
    $orig = $t
    foreach ($k in $map.Keys) { $t = $t.Replace($k, $map[$k]) }
    $t = $t -replace 'meta_data .{1,6} data_reader', "meta_data $arrow data_reader"
    if ($t -ne $orig) {
        [IO.File]::WriteAllText($_.FullName, $t, $utf8)
        $n++
    }
}
Write-Host "Patched $n files"
