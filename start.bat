@echo off
setlocal EnableExtensions

REM Qilin 后台启动（Windows）
REM - 后台静默启动后端 FastAPI（默认 hard）
REM - 后台静默启动前端 Vite 开发服务
REM - 不生成日志文件，标准输出/错误直接丢弃

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "RUN_DIR=%PROJECT_ROOT%\.qilin\run"
set "BACKEND_PID_FILE=%RUN_DIR%\backend.pid"
set "FRONTEND_PID_FILE=%RUN_DIR%\frontend.pid"

cd /d "%PROJECT_ROOT%"

if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

call "%PROJECT_ROOT%\stop.bat" >nul 2>&1

echo [Qilin] Starting backend on http://127.0.0.1:18080 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -WindowStyle Hidden -FilePath 'cmd.exe' -ArgumentList '/c','cd /d \"%PROJECT_ROOT%\" && uv run python src/backend/online/api/main.py --host 0.0.0.0 --port 18080 --tag hard >nul 2>&1' -PassThru; Set-Content -Path '%BACKEND_PID_FILE%' -Value $p.Id"

echo [Qilin] Starting frontend on http://127.0.0.1:5173 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -WindowStyle Hidden -FilePath 'cmd.exe' -ArgumentList '/c','cd /d \"%PROJECT_ROOT%\src\frontend\" && npm install --silent >nul 2>&1 && node_modules\\.bin\\vite.cmd --host 0.0.0.0 --port 5173 >nul 2>&1' -PassThru; Set-Content -Path '%FRONTEND_PID_FILE%' -Value $p.Id"

echo.
echo [Qilin] Started.
echo Backend:  http://127.0.0.1:18080/api/health
echo Frontend: http://127.0.0.1:5173
echo PID Dir:  %RUN_DIR%
