$Root = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding $false
$latin1 = [Text.Encoding]::GetEncoding('ISO-8859-1')
$fixed = 0

Get-ChildItem -Path $Root -Recurse -Include *.py,*.TXT -File | ForEach-Object {
    if ($_.FullName -match '\\scripts\\') { return }
    if ($_.FullName -notmatch 'AI_CONTEXT|object_registry|gradient_descent') {
        $peek = [IO.File]::ReadAllText($_.FullName, $utf8)
        if ($peek -notmatch 'AI_CONTEXT_START') { return }
    }
    $raw = [IO.File]::ReadAllText($_.FullName, $utf8)
    if ($raw -notmatch 'AI_CONTEXT_START') { return }
    if ($raw -notmatch 'Ã') { return }
    try {
        $bytes = $latin1.GetBytes($raw)
        $repaired = [Text.Encoding]::UTF8.GetString($bytes)
        [IO.File]::WriteAllText($_.FullName, $repaired, $utf8)
        $fixed++
    } catch {
        Write-Warning "Skip $($_.Name): $_"
    }
}
Write-Host "Repaired UTF-8 mojibake in $fixed files"
