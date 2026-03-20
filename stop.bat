@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Qilin 停止脚本（Windows）
REM - 停止后端 FastAPI（18080）
REM - 停止前端 Vite 开发服务（5173）
REM - 优先按 pid 文件停止，再按端口兜底清理

cd /d "%~dp0"

set "RUN_DIR=%CD%\.qilin\run"
set "BACKEND_PID_FILE=%RUN_DIR%\backend.pid"
set "FRONTEND_PID_FILE=%RUN_DIR%\frontend.pid"

call :stop_pid_file backend "%BACKEND_PID_FILE%"
call :stop_pid_file frontend "%FRONTEND_PID_FILE%"

call :stop_port 18080 backend
call :stop_port 5173 frontend

echo [Qilin] Stopped.
exit /b 0

:stop_pid_file
set "NAME=%~1"
set "PID_FILE=%~2"
if not exist "%PID_FILE%" exit /b 0
set /p PID=<"%PID_FILE%"
if defined PID (
  echo [Qilin] Stopping %NAME% pid=%PID%
  taskkill /PID %PID% /T /F >nul 2>&1
)
del /f /q "%PID_FILE%" >nul 2>&1
exit /b 0

:stop_port
set "PORT=%~1"
set "NAME=%~2"
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
  set "PID=%%P"
  if not "!PID!"=="0" (
    echo [Qilin] Stopping %NAME% pid=!PID! on port %PORT%
    taskkill /PID !PID! /T /F >nul 2>&1
    set "FOUND=1"
  )
)
if not defined FOUND (
  echo [Qilin] No listening process found on port %PORT%
)
exit /b 0
