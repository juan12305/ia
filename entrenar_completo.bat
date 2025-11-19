@echo off
chcp 65001 > nul
echo ================================================================
echo ENTRENAMIENTO COMPLETO - PREPARAR Y ENTRENAR
echo ================================================================
echo.

echo [PASO 1/3] Preparando FracAtlas...
echo.
py -3.12 scripts/preparar_fracatlas.py

if errorlevel 1 (
    echo.
    echo [ERROR] Fallo al preparar FracAtlas
    pause
    exit /b 1
)

echo.
echo [PASO 2/3] Fusionando datasets...
echo.
echo s | py -3.12 scripts/merge_multiple_datasets_v2.py

if errorlevel 1 (
    echo.
    echo [ERROR] Fallo en la fusion de datasets
    pause
    exit /b 1
)

echo.
echo [PASO 3/3] Iniciando entrenamiento rapido...
echo.
echo Dataset combinado: ~14,000 imagenes (2 datasets)
echo Modelo: EfficientNetB3
echo Tiempo estimado: 30-40 minutos
echo.

py -3.12 scripts/train_simple.py

echo.
echo ================================================================
echo ENTRENAMIENTO COMPLETADO
echo ================================================================
echo.
echo Modelo guardado en: models/bone_classifier_v2/best_model.pth
echo.
echo Para usar el nuevo modelo en la app web:
echo   Edita app_web.py linea 14:
echo   MODEL_PATH = "models/bone_classifier_v2/best_model.pth"
echo.

pause
