$ErrorActionPreference='Stop'
$workspaceRoot='C:\Users\amjad\Documents\Codex\2026-10-05\d'
$scientificPython=Join-Path $workspaceRoot 'work\Modeles_Generatifs\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $scientificPython)) {
    $portablePython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $portablePython) { $scientificPython=$portablePython }
    else { $scientificPython=(Get-Command python -ErrorAction Stop).Source }
}
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'Notebooks : modeles probabilistes, images latentes et sequences.'
Write-Host 'Kernel a choisir : Python 3.12 (TP modeles generatifs).'
& $scientificPython -m jupyterlab --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=10
