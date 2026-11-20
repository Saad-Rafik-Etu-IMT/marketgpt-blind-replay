#Requires -Version 5.1
<#
.SYNOPSIS
    Train research studies on a remote GPU server over SSH, with CPU fallback.
.DESCRIPTION
    The server is user@host, given with -Server or $env:MARKETGPT_GPU. Passwords are
    typed at the ssh prompt and never stored; -Setup installs an SSH key so later calls
    stop asking (if the server refuses keys, each call just asks for the password).

    Studies run with --device auto: the Transformer uses the usable GPU with the most
    free memory and falls back to CPU (reason in worker.log) when none is usable. The
    NumPy MLP always runs on CPU. After -Fetch, results appear in the local Research
    studies page like any other study.
.EXAMPLE
    .\remote-gpu.ps1 -Setup -Server user@10.0.0.5     # once: key, code, venv, torch
    $env:MARKETGPT_GPU = 'user@10.0.0.5'
    .\remote-gpu.ps1 -Probe                           # which device would be used
    .\remote-gpu.ps1 -Source SPY -Epochs 20           # push and start; prints the id
    .\remote-gpu.ps1 -Status <id>
    .\remote-gpu.ps1 -Fetch <id>
#>
[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [string]$Server = $env:MARKETGPT_GPU,
    [string]$RemoteDir = 'marketgpt-blind-replay',
    [string]$RemotePython = 'python3',

    [Parameter(ParameterSetName = 'Setup', Mandatory)][switch]$Setup,
    # cu118 wheels still ship Maxwell kernels (sm_52: Tesla M60/M40).
    [Parameter(ParameterSetName = 'Setup')]
    [string]$TorchIndex = 'https://download.pytorch.org/whl/cu118',

    [Parameter(ParameterSetName = 'Probe', Mandatory)][switch]$Probe,

    [Parameter(ParameterSetName = 'Run')][ValidatePattern('^[a-zA-Z0-9_-]{1,40}$')]
    [string]$Source = 'synthetic',
    [Parameter(ParameterSetName = 'Run')][ValidateSet('MLP', 'Transformer')]
    [string[]]$Models = @('MLP', 'Transformer'),
    [Parameter(ParameterSetName = 'Run')]
    [ValidateSet('reference', 'no-position', 'no-dropout', 'no-decay', 'fixed-lr')]
    [string[]]$Variants = @('reference'),
    [Parameter(ParameterSetName = 'Run')][int[]]$Seeds = @(0, 1, 2, 3, 4),
    [Parameter(ParameterSetName = 'Run')][ValidateRange(1, 60)][int]$Epochs = 8,
    [Parameter(ParameterSetName = 'Run')][switch]$Pooled,
    [Parameter(ParameterSetName = 'Run')][ValidatePattern('^(auto|cpu|cuda(:\d+)?)$')]
    [string]$Device = 'auto',

    [Parameter(ParameterSetName = 'Status', Mandatory)][guid]$Status,
    [Parameter(ParameterSetName = 'Fetch', Mandatory)][guid]$Fetch
)

$ErrorActionPreference = 'Stop'
if (-not $Server) { throw 'Pass -Server user@host or set $env:MARKETGPT_GPU.' }
foreach ($tool in 'ssh', 'scp') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not found. Enable the Windows OpenSSH Client optional feature."
    }
}
$backend = Join-Path $PSScriptRoot 'backend'
$studies = Join-Path $backend 'data\studies'
# Windows bsdtar: Git's GNU tar would read "C:" in a path as a remote host.
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
$upload = 'marketgpt-upload.tar.gz'

# Remote commands use single quotes only: PowerShell 5.1 mangles embedded double quotes.
function Invoke-Remote([string]$Command) {
    & ssh $Server $Command
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed (exit $LASTEXITCODE)." }
}

function Send-Backend([string[]]$Extra = @()) {
    $bundle = Join-Path ([IO.Path]::GetTempPath()) "marketgpt-$([guid]::NewGuid()).tar.gz"
    try {
        $items = @('src', 'learning', 'config', 'pyproject.toml', 'requirements.lock') + $Extra
        & $tar -czf $bundle --exclude __pycache__ --exclude '*.pyc' --exclude '*.egg-info' `
            -C $backend @items
        if ($LASTEXITCODE -ne 0) { throw 'tar failed.' }
        & scp -q $bundle "${Server}:$upload"
        if ($LASTEXITCODE -ne 0) { throw 'scp upload failed.' }
    } finally {
        Remove-Item -LiteralPath $bundle -ErrorAction SilentlyContinue
    }
    # GNU tar warns about Windows bsdtar's extended headers; those are harmless.
    # -m stamps files with the server's time, so clock skew between machines is ignored.
    return "mkdir -p ~/$RemoteDir/backend && tar -m --warning=no-unknown-keyword " +
        "--warning=no-timestamp -xzf ~/$upload -C ~/$RemoteDir/backend && rm -f ~/$upload"
}

switch ($PSCmdlet.ParameterSetName) {
    'Setup' {
        $key = Join-Path $HOME '.ssh\id_ed25519'
        if (-not (Test-Path -LiteralPath "$key.pub")) {
            Write-Host 'Creating an SSH key (press Enter twice for no passphrase).'
            & ssh-keygen -t ed25519 -f $key
            if ($LASTEXITCODE -ne 0) { throw 'ssh-keygen failed.' }
        }
        $public = (Get-Content -LiteralPath "$key.pub" -Raw).Trim()
        Write-Host 'Installing the key on the server (password prompt).'
        Invoke-Remote ("umask 077; mkdir -p ~/.ssh; grep -qxF '$public' ~/.ssh/authorized_keys " +
            "2>/dev/null || echo '$public' >> ~/.ssh/authorized_keys")

        Write-Host 'Uploading backend code.'
        $unpack = Send-Backend
        Write-Host "Creating ~/$RemoteDir/.venv and installing the backend and torch."
        # Bash in a literal here-string: PowerShell must not expand $PY or $(...).
        # Picks the first Python >= 3.10; otherwise installs one with uv into ~/.local
        # (no sudo). A .venv built from an older Python is recreated.
        $install = @'
set -e
# Ignore server-wide pip settings (e.g. an unreachable NVIDIA NGC index): this venv
# installs from PyPI and the torch index given below only.
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
ok() { command -v "$1" >/dev/null 2>&1 && "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; }
PY=
for c in __PY__ python3.13 python3.12 python3.11 python3.10 python3; do
  if ok $c; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
  echo 'No Python >= 3.10 found; installing Python 3.12 with uv into ~/.local'
  test -x ~/.local/bin/uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv python install 3.12
  PY=$(~/.local/bin/uv python find 3.12)
fi
echo "Using $PY ($($PY --version))"
cd ~/__DIR__
ok .venv/bin/python || { rm -rf .venv; $PY -m venv .venv; }
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e backend
echo 'Installing torch (about 2-3 GB, several minutes)'
# No pip cache: the CUDA wheels would otherwise take another ~3 GB of the home quota.
.venv/bin/pip install --no-cache-dir --progress-bar on torch --index-url __TORCH__
cd backend
../.venv/bin/python learning/mlp/transformer_reference.py auto
'@
        $install = $install.Replace('__PY__', $RemotePython).Replace('__DIR__', $RemoteDir).
            Replace('__TORCH__', $TorchIndex).Replace("`r`n", "`n")
        # Base64 avoids PowerShell's argument quoting and the BOM it adds to piped stdin.
        $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($install))
        # Run from a file with stdin closed: piped into bash, pip's subprocesses would
        # read (and swallow) the rest of the script, ending setup silently.
        $script = "~/.marketgpt-setup.sh"
        Invoke-Remote ("$unpack && echo $encoded | base64 -d > $script && " +
            "bash $script < /dev/null; code=`$?; rm -f $script; exit `$code")
    }
    'Probe' {
        $unpack = Send-Backend
        Invoke-Remote ("$unpack && cd ~/$RemoteDir/backend && " +
            "../.venv/bin/python learning/mlp/transformer_reference.py auto; " +
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu " +
            "--format=csv")
    }
    'Run' {
        $Seeds = @($Seeds | Select-Object -Unique)
        $Models = @($Models | Select-Object -Unique)
        $Variants = @($Variants | Select-Object -Unique)
        $id = [guid]::NewGuid().ToString()
        $spec = [ordered]@{
            source = $Source; seeds = $Seeds; models = $Models; variants = $Variants
            epochs = $Epochs; pooled = [bool]$Pooled
        }
        $extra = @("data/studies/$id")
        if ($Source -ne 'synthetic') {
            $relative = "data/raw/yfinance/$Source.csv"
            if (-not (Test-Path -LiteralPath (Join-Path $backend $relative))) {
                throw "Missing backend/$relative. Run python -m src.data.download first."
            }
            # Relative to backend/, the working directory both here and on the server.
            $spec.csv = $relative
            $extra += $(if ($Pooled) { 'data/raw/yfinance' } else { $relative })
        } elseif ($Pooled) {
            throw 'Pooled training needs a downloaded ETF source.'
        }
        $total = 0
        foreach ($model in $Models) {
            $total += $Seeds.Count * $(if ($model -eq 'Transformer') { $Variants.Count } else { 1 })
        }
        $folder = Join-Path $studies $id
        New-Item -ItemType Directory -Path $folder | Out-Null
        [IO.File]::WriteAllText((Join-Path $folder 'config.json'), ($spec | ConvertTo-Json -Compress))
        $queued = [ordered]@{ state = 'queued'; message = 'Sent to GPU server'; done = 0; total = $total }
        [IO.File]::WriteAllText((Join-Path $folder 'status.json'), ($queued | ConvertTo-Json -Compress))

        # The upload has no .git; results record the local commit the code came from.
        $commit = & git -C $PSScriptRoot rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or -not $commit) { throw 'git rev-parse HEAD failed.' }
        Write-Host "Study $id`: $total runs. Uploading code and data to $Server."
        $unpack = Send-Backend $extra
        Write-Host 'Starting the worker on the server.'
        $study = "data/studies/$id"
        # Braces background only the worker; a trailing & on the whole && chain would
        # keep the chain attached to ssh, which then waits until training ends.
        Invoke-Remote ("$unpack && cd ~/$RemoteDir/backend && { " +
            "nohup env OPENBLAS_NUM_THREADS=1 MARKETGPT_COMMIT=$commit " +
            "../.venv/bin/python learning/mlp/run_study.py " +
            "$study --device $Device > $study/worker.log 2>&1 < /dev/null & }")
        Write-Host "Started study $id on $Server (device $Device)."
        Write-Host "  .\remote-gpu.ps1 -Status $id"
        Write-Host "  .\remote-gpu.ps1 -Fetch $id"
    }
    'Status' {
        $study = "~/$RemoteDir/backend/data/studies/$Status"
        Invoke-Remote "cat $study/status.json; echo; tail -n 5 $study/worker.log"
    }
    'Fetch' {
        $temp = Join-Path ([IO.Path]::GetTempPath()) "marketgpt-$([guid]::NewGuid())"
        New-Item -ItemType Directory -Path $temp | Out-Null
        try {
            & scp -r -q "${Server}:$RemoteDir/backend/data/studies/$Fetch" $temp
            if ($LASTEXITCODE -ne 0) { throw 'scp download failed.' }
            $folder = Join-Path $studies "$Fetch"
            $keepConfig = Test-Path -LiteralPath (Join-Path $folder 'config.json')
            New-Item -ItemType Directory -Force -Path $folder | Out-Null
            Get-ChildItem -LiteralPath (Join-Path $temp "$Fetch") -File |
                Where-Object { -not ($keepConfig -and $_.Name -eq 'config.json') } |
                Copy-Item -Destination $folder -Force
            Get-Content -LiteralPath (Join-Path $folder 'status.json')
            Write-Host "`nCopied to backend/data/studies/$Fetch."
        } finally {
            Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
