@echo off
echo ================================================================
echo ENTRENAMIENTO DE MODELO - DETECTOR DE FRACTURAS
echo ================================================================
echo.

REM Detectar si hay entorno virtual
if exist "venv\Scripts\python.exe" (
    echo Detectado entorno virtual. Activando...
    call venv\Scripts\activate.bat
    python scripts/train_bone_classifier.py
) else (
    echo No se encontro entorno virtual. Usando Python 3.12 global...
    py -3.12 scripts/train_bone_classifier.py
)

pause
