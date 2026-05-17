@echo off
chcp 65001 > nul
title VisoMaster - Diagnose
cd /d "%~dp0\.."

set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
set "PY_ROOT=%~dp0\..\python"
set "SITE_PKG=%PY_ROOT%\Lib\site-packages"
set "FFMPEG_BIN=%~dp0\..\app\ffmpeg\bin"
set "PATH=%PY_ROOT%;%FFMPEG_BIN%;%SITE_PKG%\nvidia\cuda_runtime\bin;%SITE_PKG%\nvidia\cudnn\bin;%SITE_PKG%\nvidia\cublas\bin;%SITE_PKG%\nvidia\cufft\bin;%SITE_PKG%\nvidia\curand\bin;%SITE_PKG%\nvidia\cusparse\bin;%SITE_PKG%\nvidia\cusolver\bin;%SITE_PKG%\nvidia\nvjitlink\bin;%SITE_PKG%\tensorrt_libs;%SystemRoot%\System32"

echo.
echo ============================================
echo   VisoMaster Diagnose Mode (foreground log)
echo ============================================
echo.

"%PY_ROOT%\python.exe" "_internal\check_env.py"

echo.
echo --------------------------------------------
echo Launching GUI in foreground...
echo --------------------------------------------
echo.

"%PY_ROOT%\python.exe" -m _internal.launcher

pause
