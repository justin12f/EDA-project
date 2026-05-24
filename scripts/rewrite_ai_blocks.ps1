$Root = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding $false
$templatePath = Join-Path $PSScriptRoot 'blocks\statistics_ai_context.txt'
$template = [IO.File]::ReadAllText($templatePath, $utf8)

$domainLabels = @{
    descriptive = 'DescriptiveStatisticsFactory'
    inferential = 'InferentialStatisticsFactory'
    time_series = 'TimeSeriesStatisticsFactory'
    survival = 'SurvivalStatisticsFactory'
    segmentation = 'SegmentationStatisticsFactory'
    relational = 'RelationalStatisticsFactory'
    nlp = 'NlpStatisticsFactory'
    ml_support = 'MlSupportStatisticsFactory'
    graphs = 'GraphStatisticsFactory'
    geospatial = 'GeospatialStatisticsFactory'
    business = 'BusinessStatisticsFactory'
}

$pattern = '(?s)# #\[AI_CONTEXT_START\].*?# #\[AI_CONTEXT_END\]'
$n = 0

Get-ChildItem (Join-Path $Root 'statistics') -Recurse -Include *.py,*.TXT | ForEach-Object {
    $rel = $_.FullName.Substring($Root.Length + 1).Replace('\', '/')
    $domain = 'StatisticsFactory'
    if ($rel -match 'statistics/([^/]+)/') {
        $d = $Matches[1]
        if ($domainLabels.ContainsKey($d)) { $domain = $domainLabels[$d] }
    }
    $block = $template.Replace('{{FACTORY}}', $domain).TrimEnd()
    $text = [IO.File]::ReadAllText($_.FullName, $utf8)
    if ($text -notmatch 'AI_CONTEXT_START') { return }
    $newText = [regex]::Replace($text, $pattern, $block)
    if ($newText -ne $text) {
        [IO.File]::WriteAllText($_.FullName, $newText, $utf8)
        $n++
    }
}

# Non-statistics modules with bespoke blocks (UTF-8 files in blocks/)
$custom = @{
    'agents/context_creator.py' = 'agents_context.txt'
    'analyze_data/analyzers/backends/polars_impl.py' = 'analyze_polars.txt'
    'analyze_data/analyzers/backends/spark_impl.py' = 'analyze_spark.txt'
    'analyze_data/analyzers/backends/pandas_impl.py' = 'analyze_pandas.txt'
    'analyze_data/analyzers/base.py' = 'analyze_abstract.txt'
    'analyze_data/analyzers/implementations.py' = 'analyze_abstract.txt'
    'analyze_data/data_analyzer_factory.py' = 'analyze_abstract.txt'
    'readers/reader_factory.py' = 'readers.txt'
    'readers/polars_impl.py' = 'readers.txt'
    'readers/spark_impl.py' = 'readers.txt'
    'data_cleaning/data_cleaning_step_factory.py' = 'cleaning_step_factory.txt'
    'data_cleaning/data_cleaning_pipeline.py' = 'cleaning_pipeline.txt'
    'data_cleaning/data_cleaning_report.py' = 'cleaning_report.txt'
    'data_cleaning/wrapper_steps_with_logger.py' = 'cleaning_wrapper.txt'
    'model_tools/create_data_context.py' = 'model_tools.txt'
    'model_tools/data_reader_tool.py' = 'model_tools.txt'
    'model_tools/meta_data_context_tool.py' = 'model_tools.txt'
    'preproccesing/encoders/encoder_factory.py' = 'encoders.txt'
    'preproccesing/model_pre_input/prepare_input_for_context_model.py' = 'preinput.txt'
    'parsers/parser.py' = 'parser.txt'
    'evaluation/score.py' = 'evaluation.txt'
    'models/linear_regression.py' = 'linear_regression.txt'
    'gbm_config.py' = 'gbm_config.txt'
}

foreach ($rel in $custom.Keys) {
    $path = Join-Path $Root ($rel -replace '/', '\')
    if (-not (Test-Path $path)) { continue }
    $blockFile = Join-Path $PSScriptRoot ('blocks\' + $custom[$rel])
    if (-not (Test-Path $blockFile)) { continue }
    $block = [IO.File]::ReadAllText($blockFile, $utf8).TrimEnd()
    $text = [IO.File]::ReadAllText($path, $utf8)
    if ($text -notmatch 'AI_CONTEXT_START') { continue }
    $newText = [regex]::Replace($text, $pattern, $block)
    if ($newText -ne $text) {
        [IO.File]::WriteAllText($path, $newText, $utf8)
        $n++
    }
}

# data_cleaning steps share one block
$cleanBlockPath = Join-Path $PSScriptRoot 'blocks\data_cleaning_steps.txt'
if (Test-Path $cleanBlockPath) {
    $cleanBlock = [IO.File]::ReadAllText($cleanBlockPath, $utf8).TrimEnd()
    Get-ChildItem (Join-Path $Root 'data_cleaning') -Recurse -Include *.py | ForEach-Object {
        $rel = $_.FullName.Substring($Root.Length + 1).Replace('\', '/')
        if ($rel -notmatch 'data_cleaning/(steps|wrapper|pipeline|report|step_factory)') { return }
        if ($custom.ContainsKey($rel)) { return }
        $text = [IO.File]::ReadAllText($_.FullName, $utf8)
        if ($text -notmatch 'AI_CONTEXT_START') { return }
        $newText = [regex]::Replace($text, $pattern, $cleanBlock)
        if ($newText -ne $text) {
            [IO.File]::WriteAllText($_.FullName, $newText, $utf8)
            $n++
        }
    }
}

$aiRead = Join-Path $PSScriptRoot 'blocks\ai_read_global.txt'
if (Test-Path $aiRead) {
    $block = [IO.File]::ReadAllText($aiRead, $utf8).TrimEnd()
    foreach ($p in @(
        (Join-Path $Root 'statistics\AI_READ.TXT'),
        (Join-Path $Root 'data_cleaning\steps\backends\AI_READ.TXT')
    )) {
        if (Test-Path $p) {
            [IO.File]::WriteAllText($p, $block + "`n", $utf8)
            $n++
        }
    }
}

Write-Host "Rewrote blocks in $n files"
