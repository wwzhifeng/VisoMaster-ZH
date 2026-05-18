@echo off
chcp 65001 > nul
title VisoMaster TRT Launcher
cd /d "%~dp0"

REM ============================================================
REM  VisoMaster TRT Portable - main entry
REM  Double-click to launch GUI launcher
REM ============================================================

set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"

set "PY_ROOT=%~dp0python"
set "SITE_PKG=%PY_ROOT%\Lib\site-packages"

if not exist "%PY_ROOT%\python.exe" (
    echo.
    echo  [ERROR] Embedded Python not found
    echo  Expected: %PY_ROOT%\python.exe
    echo.
    echo  Make sure the pack is fully extracted, do not move files.
    echo.
    pause
    exit /b 1
)

REM Build DLL+ffmpeg search path within this process only (all relative to pack root)
set "FFMPEG_BIN=%~dp0app\ffmpeg\bin"
set "PATH=%PY_ROOT%;%FFMPEG_BIN%;%SITE_PKG%\nvidia\cuda_runtime\bin;%SITE_PKG%\nvidia\cudnn\bin;%SITE_PKG%\nvidia\cublas\bin;%SITE_PKG%\nvidia\cufft\bin;%SITE_PKG%\nvidia\curand\bin;%SITE_PKG%\nvidia\cusparse\bin;%SITE_PKG%\nvidia\cusolver\bin;%SITE_PKG%\nvidia\nvjitlink\bin;%SITE_PKG%\tensorrt_libs;%SystemRoot%\System32;%SystemRoot%"

REM Launch GUI in background (pythonw = no console)
start "" "%PY_ROOT%\pythonw.exe" -m _internal.launcher

REM Fallback: if pythonw silently dies, show error in foreground
timeout /t 2 /nobreak > nul
tasklist /fi "imagename eq pythonw.exe" 2>nul | find /i "pythonw.exe" >nul
if errorlevel 1 (
    echo.
    echo  [WARN] Background launch failed, switching to foreground mode...
    echo.
    "%PY_ROOT%\python.exe" -m _internal.launcher
    if errorlevel 1 (
        echo.
        echo  Launcher failed. Run _internal\diagnose.bat for details.
        echo  See README.txt troubleshooting section.
        pause
    )
)
exit /b 0
