# ============================================================
#  VisoMaster TRT Portable - Build Script
#  Run on a clean Windows machine to assemble the integration pack
# ------------------------------------------------------------
#  Usage:
#    PS> cd D:\GitHub\VisoMaster-TRT-Portable
#    PS> powershell -ExecutionPolicy Bypass -File .\_internal\_build_package.ps1 -Stage python
#
#  Stages:
#    -Stage python      Embedded Python + pip bootstrap
#    -Stage deps        Install pip dependencies (cu128)
#    -Stage source      Clone VisoMaster source and apply patches
#    -Stage assets      Verify scripts/configs exist
#    -Stage pack        7z the result
#    -Stage all         Run all stages in order
# ============================================================
param(
    [ValidateSet("all","python","deps","source","assets","pack")]
    [string]$Stage = "all",
    [string]$WorkDir = "",
    [string]$PyVersion = "3.10.11",
    [string]$VisoMasterRef = "main",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrEmpty($WorkDir)) {
    $WorkDir = Split-Path -Parent $PSScriptRoot
}
$WorkDir = (Resolve-Path $WorkDir).Path
$PyRoot  = Join-Path $WorkDir "python"
$AppRoot = Join-Path $WorkDir "app"
$Patches = Join-Path $WorkDir "_internal\patches"

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Info($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
function Ok($msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Err($msg)  { Write-Host "    [!!] $msg" -ForegroundColor Red }


# ------------------------------------------------------------
# Stage 1: Embedded Python
# ------------------------------------------------------------
function Stage-Python {
    Step "Stage: Embedded Python $PyVersion"

    $zipUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
    $zipPath = Join-Path $env:TEMP "python-$PyVersion-embed.zip"

    if ((Test-Path "$PyRoot\python.exe") -and -not $Force) {
        Info "Already exists at $PyRoot\python.exe (use -Force to reinstall)"
    } else {
        Info "Downloading $zipUrl"
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        Info "Extracting to $PyRoot"
        Get-ChildItem -Path $PyRoot -Filter "*.dll" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        Expand-Archive -Path $zipPath -DestinationPath $PyRoot -Force
        Ok "Python extracted"
    }

    # Write python310._pth (enable site)
    Info "Writing python310._pth"
    $pthLines = @(
        "python310.zip",
        ".",
        "..",
        "..\app",
        ".\Lib\site-packages",
        "",
        "import site"
    )
    $pthLines -join "`r`n" | Out-File -FilePath "$PyRoot\python310._pth" -Encoding ASCII -NoNewline

    # Bootstrap pip
    if (-not (Test-Path "$PyRoot\Lib\site-packages\pip")) {
        Info "Bootstrapping pip"
        $getPip = Join-Path $env:TEMP "get-pip.py"
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip -UseBasicParsing
        & "$PyRoot\python.exe" $getPip --no-warn-script-location
        if ($LASTEXITCODE -ne 0) { Err "pip bootstrap failed"; exit 1 }
        Ok "pip installed"
    } else {
        Info "pip already present, skipping"
    }

    Ok "Stage python complete"
}


# ------------------------------------------------------------
# Stage 2: Pip dependencies
# ------------------------------------------------------------
function Stage-Deps {
    Step "Stage: pip dependencies (cu128)"

    $req = Join-Path $WorkDir "_internal\requirements_portable_cu128.txt"
    if (-not (Test-Path $req)) {
        Err "Missing $req"; exit 1
    }

    Info "Upgrading pip / setuptools / wheel"
    & "$PyRoot\python.exe" -m pip install --upgrade pip setuptools wheel --no-warn-script-location

    Info "Installing requirements from PyTorch cu128 + NVIDIA indexes"
    & "$PyRoot\python.exe" -m pip install -r $req `
        --extra-index-url "https://download.pytorch.org/whl/cu128" `
        --extra-index-url "https://pypi.nvidia.com" `
        --no-warn-script-location

    if ($LASTEXITCODE -ne 0) { Err "pip install failed"; exit 1 }
    Ok "Dependencies installed"

    Info "Purging pip cache to shrink size"
    & "$PyRoot\python.exe" -m pip cache purge | Out-Null
    Ok "Stage deps complete"
}


# ------------------------------------------------------------
# Stage 3: VisoMaster source + patches
# ------------------------------------------------------------
function Stage-Source {
    Step "Stage: clone VisoMaster source"

    $tmpClone = Join-Path $env:TEMP "VisoMaster-src"
    if (Test-Path $tmpClone) { Remove-Item -Recurse -Force $tmpClone }

    Info "git clone (ref: $VisoMasterRef)"
    git clone --depth 1 --branch $VisoMasterRef `
        https://github.com/visomaster/VisoMaster.git $tmpClone
    if (-not (Test-Path "$tmpClone\main.py")) { Err "git clone failed"; exit 1 }

    Info "Copying source to app\"
    if (Test-Path $AppRoot) { Remove-Item -Recurse -Force $AppRoot }
    New-Item -ItemType Directory -Force $AppRoot | Out-Null
    Copy-Item -Recurse -Force "$tmpClone\app\*" -Destination $AppRoot

    # Move main.py to project root (bootstrap.py expects it there)
    Copy-Item -Force "$tmpClone\main.py" "$WorkDir\main.py"

    # Also copy download_models.py for later URL extraction
    if (Test-Path "$tmpClone\download_models.py") {
        Copy-Item -Force "$tmpClone\download_models.py" "$WorkDir\_internal\_upstream_download_models.py"
    }

    # Apply patches if any
    if (Test-Path "$Patches\*.patch") {
        Info "Applying patches"
        Push-Location $WorkDir
        Get-ChildItem "$Patches\*.patch" | ForEach-Object {
            Info "  patch: $($_.Name)"
            git apply --whitespace=fix $_.FullName
            if ($LASTEXITCODE -ne 0) { Err "patch failed: $($_.Name)"; exit 1 }
        }
        Pop-Location
        Ok "All patches applied"
    } else {
        Info "No .patch files in _internal\patches\, skipping (using runtime monkey-patches instead)"
    }

    Remove-Item -Recurse -Force $tmpClone
    Ok "Stage source complete"
}


# ------------------------------------------------------------
# Stage 4: Assets check
# ------------------------------------------------------------
function Stage-Assets {
    Step "Stage: verify scripts and configs"
    $required = @(
        "Start.bat",
        "config.ini", "README.txt",
        "_internal\bootstrap.py", "_internal\check_env.py",
        "_internal\build_engines.py", "_internal\model_manager.py",
        "_internal\manifest.json",
        "_internal\launcher\main_window.py",
        "_internal\launcher\workers.py",
        "_internal\launcher\dialogs.py"
    )
    $missing = @()
    foreach ($f in $required) {
        if (-not (Test-Path (Join-Path $WorkDir $f))) {
            $missing += $f
        }
    }
    if ($missing.Count -gt 0) {
        Err "Missing files:"
        $missing | ForEach-Object { Err "  - $_" }
        exit 1
    }
    Ok "All scripts and configs present"
}


# ------------------------------------------------------------
# Stage 5: 7z package
# ------------------------------------------------------------
function Stage-Pack {
    Step "Stage: 7z packaging"

    $sevenZip = (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source
    if (-not $sevenZip) {
        Err "7z.exe not found in PATH. Install 7-Zip and try again."
        exit 1
    }

    $outDir = Join-Path $WorkDir "..\dist"
    New-Item -ItemType Directory -Force $outDir | Out-Null

    $stamp = Get-Date -Format "yyyyMMdd"
    $mainOut = Join-Path $outDir "VisoMaster-TRT-Portable-cu128-$stamp.7z"
    $modelsOut = Join-Path $outDir "VisoMaster-Models-Full-$stamp.7z"

    Info "Packing main archive -> $mainOut"
    Push-Location $WorkDir
    & $sevenZip a -mx=9 -ms=on -mmt=on -t7z $mainOut `
        "python\" "app\" "_internal\" "workspace\" "output\" "tmp\" `
        "main.py" "Start.bat" "config.ini" "README.txt" `
        "-xr!__pycache__" "-xr!*.pyc" "-xr!*.pyo"
    Pop-Location
    Ok "Main package: $mainOut"

    $modelFiles = Get-ChildItem -Path "$WorkDir\models" -Recurse -Filter "*.onnx" -ErrorAction SilentlyContinue
    if ($modelFiles.Count -gt 0) {
        Info "Packing models archive -> $modelsOut"
        Push-Location $WorkDir
        & $sevenZip a -mx=9 -t7z $modelsOut "models\"
        Pop-Location
        Ok "Models package: $modelsOut"
    } else {
        Info "models\ is empty, skipping models package"
    }

    Step "Done"
    Get-ChildItem $outDir | Format-Table Name, @{N="MB";E={[math]::Round($_.Length/1MB,1)}}
}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  VisoMaster TRT Portable - Builder" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  Stage:   $Stage"
Write-Host "  WorkDir: $WorkDir"
Write-Host ""

switch ($Stage) {
    "python" { Stage-Python }
    "deps"   { Stage-Deps }
    "source" { Stage-Source }
    "assets" { Stage-Assets }
    "pack"   { Stage-Pack }
    "all" {
        Stage-Python
        Stage-Deps
        Stage-Source
        Stage-Assets
        Stage-Pack
    }
}

Write-Host ""
Write-Host "Build done." -ForegroundColor Green
