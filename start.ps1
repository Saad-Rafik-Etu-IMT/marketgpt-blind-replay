#Requires -Version 5.1
<#
.SYNOPSIS
    Start the local replay backend and frontend using the target environment.
.PARAMETER Install
    Create the local environment and install locked dependencies, then exit.
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$backend = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'
$python = Join-Path $root '.venv\Scripts\python.exe'
$npm = Get-Command 'npm.cmd' -ErrorAction SilentlyContinue
if (-not $npm) { throw 'npm was not found. Install Node.js >= 18.' }

if ($Install) {
    if (-not (Test-Path -LiteralPath $python)) {
        $launcher = Get-Command 'python' -ErrorAction SilentlyContinue
        if (-not $launcher) { $launcher = Get-Command 'py' -ErrorAction SilentlyContinue }
        if (-not $launcher) { throw 'Python was not found. Install Python >= 3.10.' }
        & $launcher.Source -m venv (Join-Path $root '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the target environment failed.' }
    }
    & $python -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Bootstrapping pip failed.' }
    Push-Location $backend
    try {
        & $python -m pip install -c requirements.lock -e '.[dev]'
        if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
    } finally { Pop-Location }
    Push-Location $frontend
    try {
        & $npm.Source ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    } finally { Pop-Location }
    Write-Host 'Dependencies installed. Run .\start.ps1 to launch.'
    return
}

if (-not (Test-Path -LiteralPath $python) -or
    -not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules'))) {
    throw 'Dependencies are missing. Run .\start.ps1 -Install.'
}
& $python -c 'import uvicorn, fastapi'
if ($LASTEXITCODE -ne 0) { throw 'Backend dependencies are missing. Run .\start.ps1 -Install.' }

function Stop-OwnedTree([int]$ProcessId) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId"
    foreach ($child in $children) { Stop-OwnedTree -ProcessId $child.ProcessId }
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

$oldOrigin = $env:MRTL_FRONTEND_ORIGIN
$oldApi = $env:NEXT_PUBLIC_API_BASE_URL
$procs = @()
try {
    $env:MRTL_FRONTEND_ORIGIN = "http://localhost:$FrontendPort"
    $env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:$BackendPort"
    Write-Host "Backend: http://localhost:$BackendPort/docs"
    Write-Host "Frontend: http://localhost:$FrontendPort"
    Write-Host 'No data is bundled. Upload through the API docs. Ctrl+C stops both services.'
    $be = Start-Process -FilePath $python `
        -ArgumentList '-m', 'uvicorn', 'src.main:app', '--port', $BackendPort, '--reload' `
        -WorkingDirectory $backend -NoNewWindow -PassThru
    $procs += $be
    $fe = Start-Process -FilePath $npm.Source `
        -ArgumentList 'run', 'dev', '--', '-p', $FrontendPort `
        -WorkingDirectory $frontend -NoNewWindow -PassThru
    $procs += $fe
    while (-not $be.HasExited -and -not $fe.HasExited) {
        Start-Sleep -Milliseconds 500
    }
    if ($be.HasExited) { throw "Backend exited with code $($be.ExitCode)." }
    if ($fe.HasExited) { throw "Frontend exited with code $($fe.ExitCode)." }
} finally {
    foreach ($proc in $procs) { Stop-OwnedTree -ProcessId $proc.Id }
    $env:MRTL_FRONTEND_ORIGIN = $oldOrigin
    $env:NEXT_PUBLIC_API_BASE_URL = $oldApi
}
