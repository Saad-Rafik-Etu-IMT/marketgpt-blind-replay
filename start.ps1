#Requires -Version 5.1
<#
.SYNOPSIS
    Start the local replay backend and frontend using the target environment.
    Works on Windows (PowerShell 5.1 or 7+) and on Linux/macOS (PowerShell 7+).
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

# Windows PowerShell 5.1 is always Windows and has no $IsWindows variable.
$onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or [bool]$IsWindows
if ($onWindows) {
    $python = Join-Path $root '.venv\Scripts\python.exe'
    $npmName = 'npm.cmd'
    $launcherNames = @('python', 'py')
    $startHint = '.\start.ps1'
} else {
    $python = Join-Path $root '.venv/bin/python'
    $npmName = 'npm'
    $launcherNames = @('python3', 'python')
    $startHint = './start.ps1'
}

$npm = Get-Command $npmName -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $npm) { throw 'npm was not found. Install Node.js >= 18.' }

if ($Install) {
    if (-not (Test-Path -LiteralPath $python)) {
        $launcher = $null
        foreach ($name in $launcherNames) {
            $launcher = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($launcher) { break }
        }
        if (-not $launcher) { throw 'Python was not found. Install Python >= 3.10.' }
        & $launcher.Source -m venv (Join-Path $root '.venv')
        if ($LASTEXITCODE -ne 0) {
            $venvHint = if ($onWindows) { '' } else { ' On Debian/Ubuntu, install the python3-venv package.' }
            throw "Creating the target environment failed.$venvHint"
        }
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
    Write-Host "Dependencies installed. Run $startHint to launch."
    return
}

if (-not (Test-Path -LiteralPath $python) -or
    -not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules'))) {
    throw "Dependencies are missing. Run $startHint -Install."
}
& $python -c 'import uvicorn, fastapi'
if ($LASTEXITCODE -ne 0) { throw "Backend dependencies are missing. Run $startHint -Install." }

# Direct children of a process: CIM on Windows, pgrep on Linux/macOS.
function Get-ChildProcessIds([int]$ProcessId) {
    if ($onWindows) {
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId"
        foreach ($child in $children) { [int]$child.ProcessId }
    } elseif (Get-Command 'pgrep' -CommandType Application -ErrorAction SilentlyContinue) {
        $ids = & pgrep -P $ProcessId 2>$null
        foreach ($id in $ids) { if ($id) { [int]$id } }
    }
}

function Stop-OwnedTree([int]$ProcessId) {
    foreach ($childId in (Get-ChildProcessIds -ProcessId $ProcessId)) {
        Stop-OwnedTree -ProcessId $childId
    }
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
