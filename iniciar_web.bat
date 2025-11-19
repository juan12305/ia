@echo off
echo ================================================================
echo DETECTOR DE FRACTURAS - APLICACION WEB
echo ================================================================
echo.

REM Instalar Gradio si no está instalado
echo Verificando Gradio...
py -3.12 -m pip show gradio >nul 2>&1
if errorlevel 1 (
    echo Instalando Gradio...
    py -3.12 -m pip install gradio
    echo.
)

echo Iniciando aplicacion web...
echo.
echo La aplicacion se abrira en tu navegador en:
echo   http://localhost:7860
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

py -3.12 app_web.py

pause
