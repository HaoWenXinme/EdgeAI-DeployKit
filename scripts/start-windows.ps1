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
$RuntimeDir = Join-Path $Root ".runtime"
New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null
$env:COREPACK_ENABLE_DOWNLOAD_PROMPT = "0"

function Add-PathDir([string]$Dir) {
  if ($Dir -and (Test-Path $Dir)) {
    $env:PATH = "$Dir;$env:PATH"
  }
}

function Add-LocalNodeRuntime {
  $nodeRoot = Join-Path $Root ".runtime"
  if (-not (Test-Path $nodeRoot)) {
    return
  }
  $localNode = Get-ChildItem -Path $nodeRoot -Directory -Filter "node-v*-win-x64" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1
  if ($localNode -and (Test-Path (Join-Path $localNode.FullName "node.exe"))) {
    Add-PathDir $localNode.FullName
  }
}

function Install-PortableNodeRuntime {
  if ($NoInstall) {
    return
  }
  $version = if ($env:EDGEAI_NODE_VERSION) { $env:EDGEAI_NODE_VERSION } else { "v22.17.1" }
  if (-not $version.StartsWith("v")) {
    $version = "v$version"
  }
  $folderName = "node-$version-win-x64"
  $nodeDir = Join-Path $RuntimeDir $folderName
  $nodeExe = Join-Path $nodeDir "node.exe"
  if (Test-Path $nodeExe) {
    Add-PathDir $nodeDir
    return
  }

  $baseUrl = if ($env:NODE_DIST_URL) { $env:NODE_DIST_URL.TrimEnd("/") } else { "https://nodejs.org/dist" }
  $url = "$baseUrl/$version/$folderName.zip"
  $zipPath = Join-Path $RuntimeDir "$folderName.zip"
  New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
  Write-Host "[EdgeAI] Node.js/Corepack not found. Downloading portable Node.js $version..."
  Write-Host "[EdgeAI] URL: $url"
  try {
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    Expand-Archive -Path $zipPath -DestinationPath $RuntimeDir -Force
  } catch {
    throw @"
Node.js was not found, and portable Node.js could not be downloaded.

Please install Node.js 20+ / 22+ LTS, or set NODE_DIST_URL to a reachable mirror, then rerun start-windows.bat.
Example mirror:
  set NODE_DIST_URL=https://npmmirror.com/mirrors/node

Original error: $($_.Exception.Message)
"@
  } finally {
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
  }
  if (-not (Test-Path $nodeExe)) {
    throw "Portable Node.js extraction finished but node.exe was not found: $nodeExe"
  }
  Add-PathDir $nodeDir
}

function Ensure-NodeRuntime {
  Add-LocalNodeRuntime
  if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Install-PortableNodeRuntime
  }
  if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js 20+ / 22+ was not found. Install Node.js LTS or rerun with network access so portable Node.js can be downloaded."
  }
  if (-not (Get-Command pnpm -ErrorAction SilentlyContinue) -and -not (Get-Command corepack -ErrorAction SilentlyContinue)) {
    Install-PortableNodeRuntime
  }
  if (-not (Get-Command pnpm -ErrorAction SilentlyContinue) -and -not (Get-Command corepack -ErrorAction SilentlyContinue)) {
    throw "pnpm/Corepack was not found even after preparing Node.js. Install Node.js LTS and rerun start-windows.bat."
  }
}

function Find-LlamaCliRuntime {
  $envCli = $env:EDGEAI_LLAMA_CLI
  if ($envCli -and (Test-Path $envCli)) {
    return (Resolve-Path $envCli).Path
  }
  $localCli = Get-ChildItem -Path (Join-Path $RuntimeDir "llama.cpp") -Recurse -Filter "llama-cli.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($localCli) {
    return $localCli.FullName
  }
  $pathCli = Get-Command llama-cli -ErrorAction SilentlyContinue
  if ($pathCli) {
    return $pathCli.Source
  }
  $pathLegacy = Get-Command llama -ErrorAction SilentlyContinue
  if ($pathLegacy) {
    return $pathLegacy.Source
  }
  return $null
}

function Install-LlamaCppRuntime {
  if ($NoInstall) {
    return
  }
  $llamaRoot = Join-Path $RuntimeDir "llama.cpp"
  $llamaBin = Join-Path $llamaRoot "bin"
  $llamaCli = Join-Path $llamaBin "llama-cli.exe"
  if (Test-Path $llamaCli) {
    $env:EDGEAI_LLAMA_CLI = $llamaCli
    Add-PathDir $llamaBin
    return
  }

  $tag = if ($env:EDGEAI_LLAMA_CPP_TAG) { $env:EDGEAI_LLAMA_CPP_TAG } else { "b9787" }
  $asset = "llama-$tag-bin-win-cpu-x64.zip"
  $url = if ($env:EDGEAI_LLAMA_CPP_URL) { $env:EDGEAI_LLAMA_CPP_URL } else { "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$asset" }
  $zipPath = Join-Path $llamaRoot $asset
  New-Item -ItemType Directory -Force -Path $llamaRoot, $llamaBin | Out-Null
  Write-Host "[EdgeAI] GGUF runtime not found. Downloading llama.cpp CPU runtime..."
  Write-Host "[EdgeAI] URL: $url"
  try {
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    Expand-Archive -Path $zipPath -DestinationPath $llamaBin -Force
  } catch {
    throw @"
GGUF runtime could not be downloaded.

Install llama.cpp manually and set EDGEAI_LLAMA_CLI to llama-cli.exe, or set EDGEAI_LLAMA_CPP_URL to a reachable zip mirror.
Original error: $($_.Exception.Message)
"@
  } finally {
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
  }

  $found = Find-LlamaCliRuntime
  if (-not $found) {
    throw "llama.cpp extraction finished but llama-cli.exe was not found under $llamaRoot"
  }
  $env:EDGEAI_LLAMA_CLI = $found
  Add-PathDir (Split-Path -Parent $found)
}

function Ensure-LlamaRuntime {
  $found = Find-LlamaCliRuntime
  if (-not $found) {
    Install-LlamaCppRuntime
    $found = Find-LlamaCliRuntime
  }
  if ($found) {
    $env:EDGEAI_LLAMA_CLI = $found
    Add-PathDir (Split-Path -Parent $found)
    Write-Host "[EdgeAI] GGUF runtime: $found"
  } elseif ($WithLLM) {
    throw "GGUF runtime was requested but llama-cli.exe was not found."
  } else {
    Write-Host "[EdgeAI] GGUF runtime not configured. Use -WithLLM to enable local GGUF chat."
  }
}

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
  throw @"
Python 3.9-3.13 was not found.

Please install Python 3.10/3.11/3.12 and enable "Add python.exe to PATH", then rerun start-windows.bat.
You can also double-click install-runtime-windows.bat to try installing Python with winget.
"@
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
  Ensure-NodeRuntime
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
    Ensure-LlamaRuntime
  } else {
    Ensure-LlamaRuntime
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

Ensure-NodeRuntime
Ensure-LlamaRuntime

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
$escapedPath = $env:PATH.Replace("'", "''")
$uiCommand = @"
Set-Location '$ProductUi'
`$env:PATH = '$escapedPath'
`$env:NEXT_PUBLIC_API_BASE = '$ApiBase'
if (Test-Path '.\node_modules\next\dist\bin\next') {
  & node .\node_modules\next\dist\bin\next dev --hostname 127.0.0.1 --port $UiPort
} elseif (Get-Command pnpm -ErrorAction SilentlyContinue) {
  & pnpm dev --hostname 127.0.0.1 --port $UiPort
} else {
  & corepack pnpm dev --hostname 127.0.0.1 --port $UiPort
}
"@
$uiProcess = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $uiCommand) -WorkingDirectory $ProductUi -RedirectStandardOutput $UiOut -RedirectStandardError $UiErr -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $PidDir "webui.pid") -Value $uiProcess.Id
Set-Content -Path (Join-Path $PidDir "webui.port") -Value $UiPort

Write-Host "[EdgeAI] started"
Write-Host "[EdgeAI] open: $UiUrl"

if (-not $NoBrowser) {
  Start-Process $UiUrl | Out-Null
}
