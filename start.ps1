[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 5188,
    [string]$DatasetPath,
    [string]$DataDir,
    [string]$PythonPath = (Join-Path $env:USERPROFILE 'miniconda3\envs\torch_env\python.exe')
)

$ErrorActionPreference = 'Stop'
if (-not $DatasetPath) {
    $desktop = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $DatasetPath = Join-Path $desktop 'new_dataset'
}
if (-not $DataDir) {
    $DataDir = Join-Path $PSScriptRoot '.local'
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python was not found at $PythonPath. Set -PythonPath to the torch_env interpreter."
}
& $PythonPath -B (Join-Path $PSScriptRoot 'server.py') --port $Port --dataset $DatasetPath --data-dir $DataDir
if ($LASTEXITCODE -ne 0) {
    throw "The workbench exited with code $LASTEXITCODE."
}
