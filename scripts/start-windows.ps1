param(
  [int]$ApiPort = 8000,
  [int]$UiPort = 3000,
  [switch]$NoInstall,
  [switch]$NoFrontendInstall,
  [switch]$WithPytorch,
  [switch]$WithTensorflow,
  [switch]$WithML,
  [switch]$WithLLM,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Split-Path -Leaf $ScriptDir) -ieq "scripts") {
  $Root = Split-Path -Parent $ScriptDir
} else {
  $Root = $ScriptDir
}

$LogDir = Join-Path $Root "outputs\logs"
$PidDir = Join-Path $Root "outputs\pids"
New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null
$env:COREPACK_ENABLE_DOWNLOAD_PROMPT = "0"

function Find-Python {
  if ($env:PYTHON_BIN) {
    return @{ Exe = $env:PYTHON_BIN; Args = @() }
  }
  $candidates = @()
  if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyList = & py -0p 2>$null
    foreach ($line in $pyList) {
      if ($line -match "^\s+-V:[^\s]+\s+(.+?python\.exe)\s*$") {
        $candidates += $Matches[1]
      }
    }
  }
  if (Get-Command python -ErrorAction SilentlyContinue) {
    $candidates += "python"
  }

  $valid = @()
  foreach ($candidate in ($candidates | Select-Object -Unique)) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $version = & $candidate -c "import sys, site; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($code -ne 0 -or -not $version) {
      continue
    }
    $parts = [string]$version -split "\."
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    if ($major -ne 3 -or $minor -lt 9 -or $minor -gt 13) {
      continue
    }
    $rank = switch ($minor) {
      12 { 0 }
      11 { 1 }
      10 { 2 }
      9 { 3 }
      13 { 4 }
      default { 99 }
    }
    $valid += [pscustomobject]@{ Exe = $candidate; Rank = $rank; Version = $version }
  }

  $chosen = $valid | Sort-Object Rank, Version | Select-Object -First 1
  if ($chosen) {
    Write-Host "[EdgeAI] python: $($chosen.Exe) $($chosen.Version)"
    return @{ Exe = $chosen.Exe; Args = @() }
  }
  throw "Python 3.9-3.13 was not found. Install a supported Python version, then rerun start-windows.bat."
}

function Invoke-Python($PythonSpec, [string[]]$Arguments) {
  & $PythonSpec.Exe @($PythonSpec.Args + $Arguments)
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed: $($Arguments -join ' ')"
  }
}

function Get-FreePort([int]$PreferredPort) {
  $port = $PreferredPort
  while ($true) {
    $listener = $null
    try {
      $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
      $listener.Start()
      return $port
    } catch {
      $port += 1
    } finally {
      if ($listener) {
        $listener.Stop()
      }
    }
  }
}

function Invoke-Pnpm([string[]]$Arguments) {
  if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    & pnpm @Arguments
  } elseif (Get-Command corepack -ErrorAction SilentlyContinue) {
    & corepack pnpm @Arguments
  } else {
    throw "pnpm/Corepack was not found. Install Node.js 20+ with Corepack, then rerun start-windows.bat."
  }
  if ($LASTEXITCODE -ne 0) {
    throw "pnpm command failed: $($Arguments -join ' ')"
  }
}

Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$InstallMarker = Join-Path $Root ".venv\.edgeai-windows-installed"

if (-not $NoInstall -and -not (Test-Path $InstallMarker)) {
  $PythonSpec = Find-Python
  Write-Host "[EdgeAI] creating/updating Python environment..."
  if (-not (Test-Path $VenvPython)) {
    Invoke-Python $PythonSpec @("-m", "venv", (Join-Path $Root ".venv"))
  }
  & $VenvPython -m pip install --upgrade pip setuptools wheel
  if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed." }
  & $VenvPython -m pip install -e ".[pdf]"
  if ($LASTEXITCODE -ne 0) { throw "EdgeAI Python dependency install failed." }

  if ($WithPytorch) {
    $pytorchIndex = if ($env:PYTORCH_INDEX_URL) { $env:PYTORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cpu" }
    & $VenvPython -m pip install --index-url $pytorchIndex torch torchvision
    if ($LASTEXITCODE -ne 0) { throw "PyTorch dependency install failed." }
  } else {
    Write-Host "[EdgeAI] PyTorch conversion deps skipped. Use -WithPytorch to enable .pt/.pth conversion."
  }

  if ($WithTensorflow) {
    & $VenvPython -m pip install tensorflow-cpu tf2onnx h5py
    if ($LASTEXITCODE -ne 0) { throw "TensorFlow dependency install failed." }
  } else {
    Write-Host "[EdgeAI] TensorFlow conversion deps skipped. Use -WithTensorflow to enable .h5/.keras conversion."
  }

  if ($WithML) {
    & $VenvPython -m pip install ".[traditional-ml]"
    if ($LASTEXITCODE -ne 0) { throw "Traditional ML dependency install failed." }
  } else {
    Write-Host "[EdgeAI] Traditional ML deps skipped. Use -WithML to enable sklearn/xgboost/lightgbm conversion."
  }

  if ($WithLLM) {
    & $VenvPython -m pip install ".[llm]"
    if ($LASTEXITCODE -ne 0) { throw "LLM runtime dependency install failed." }
  } else {
    Write-Host "[EdgeAI] LLM runtime deps skipped. Use -WithLLM or install llama.cpp to enable GGUF chat."
  }

  if (-not $NoFrontendInstall) {
    Write-Host "[EdgeAI] installing WebUI dependencies..."
    Set-Location (Join-Path $Root "product-ui")
    if (Get-Command corepack -ErrorAction SilentlyContinue) {
      & corepack enable 2>$null
    }
    Invoke-Pnpm @("install", "--frozen-lockfile")
    Set-Location $Root
  }

  New-Item -ItemType File -Force -Path $InstallMarker | Out-Null
}

if (-not (Test-Path $VenvPython)) {
  throw "Missing .venv. Run start-windows.bat without -NoInstall first."
}

$ApiPort = Get-FreePort $ApiPort
$UiPort = Get-FreePort $UiPort
$ApiBase = "http://127.0.0.1:$ApiPort"
$UiUrl = "http://127.0.0.1:$UiPort/workspace"

Write-Host "[EdgeAI] project root: $Root"
Write-Host "[EdgeAI] API:  $ApiBase"
Write-Host "[EdgeAI] UI:   $UiUrl"
Write-Host "[EdgeAI] logs: $LogDir"

$ApiOut = Join-Path $LogDir "api.log"
$ApiErr = Join-Path $LogDir "api.err.log"
$apiArgs = @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "$ApiPort")
$apiProcess = Start-Process -FilePath $VenvPython -ArgumentList $apiArgs -WorkingDirectory $Root -RedirectStandardOutput $ApiOut -RedirectStandardError $ApiErr -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $PidDir "api.pid") -Value $apiProcess.Id
Set-Content -Path (Join-Path $PidDir "api.port") -Value $ApiPort

$healthOk = $false
for ($i = 0; $i -lt 45; $i++) {
  try {
    Invoke-WebRequest -Uri "$ApiBase/api/health" -UseBasicParsing -TimeoutSec 1 | Out-Null
    $healthOk = $true
    break
  } catch {
    Start-Sleep -Seconds 1
  }
}
if (-not $healthOk) {
  throw "Backend did not become healthy. Check $ApiOut and $ApiErr."
}

$UiOut = Join-Path $LogDir "ui.log"
$UiErr = Join-Path $LogDir "ui.err.log"
$ProductUi = Join-Path $Root "product-ui"
$pnpmCommand = if (Get-Command pnpm -ErrorAction SilentlyContinue) { "pnpm" } else { "corepack pnpm" }
$uiCommand = @"
Set-Location '$ProductUi'
`$env:NEXT_PUBLIC_API_BASE = '$ApiBase'
& $pnpmCommand dev --hostname 127.0.0.1 --port $UiPort
"@
$uiProcess = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $uiCommand) -WorkingDirectory $ProductUi -RedirectStandardOutput $UiOut -RedirectStandardError $UiErr -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $PidDir "webui.pid") -Value $uiProcess.Id
Set-Content -Path (Join-Path $PidDir "webui.port") -Value $UiPort

Write-Host "[EdgeAI] started"
Write-Host "[EdgeAI] open: $UiUrl"

if (-not $NoBrowser) {
  Start-Process $UiUrl | Out-Null
}
