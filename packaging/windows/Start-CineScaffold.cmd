@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-CineScaffold.ps1"
if errorlevel 1 (
  echo.
  echo CineScaffold 启动失败，请保留本窗口并把错误信息发给开发者。
  pause
  exit /b 1
)
