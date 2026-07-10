@echo off
REM Wi-Fi Radar — un click para arrancar todo.
REM Abre 3 ventanas de PowerShell: FastAPI, CSI reader, Camera. Luego el browser.

cd /d "%~dp0"

echo.
echo ================================================
echo   Wi-Fi Radar  -  Iniciando sistema completo
echo ================================================
echo.
echo Abriendo 3 procesos en ventanas separadas:
echo   1. FastAPI backend (dashboard 3D)
echo   2. ESP32 CSI reader (motion en COM7)
echo   3. Camara con face ID
echo.

REM 1) FastAPI backend
start "Radar - FastAPI" powershell -NoExit -Command "Write-Host '[1/3] FastAPI backend' -ForegroundColor Cyan; python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000"

REM Wait for backend to be up before starting clients
timeout /t 4 /nobreak >nul

REM 2) ESP32 CSI reader (filtra al BSSID de tu router para signal limpio)
start "Radar - ESP32 CSI" powershell -NoExit -Command "Write-Host '[2/3] ESP32 CSI radar' -ForegroundColor Magenta; python -m backend.csi_reader live --port COM7 --broadcast http://127.0.0.1:8000 --source-mac aa:bb:cc:dd:ee:ff"

REM 3) Camera with face ID
start "Radar - Camara" powershell -NoExit -Command "Write-Host '[3/3] Camara YOLO + Face ID' -ForegroundColor Green; python -m backend.camera_detector radar --listen http://127.0.0.1:8000 --broadcast http://127.0.0.1:8000"

REM Open the browser
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8000

echo.
echo Listo. El navegador deberia abrirse en unos segundos.
echo Para detener todo: cierra cada ventana de PowerShell con Ctrl+C
echo  y la ventana de la camara con 'q'.
echo.
pause
