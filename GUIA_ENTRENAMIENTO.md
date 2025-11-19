# 🚀 Guía de Entrenamiento Rápido

## 📊 Datasets Incluidos

1. **Bone_Fracture_Binary_Classification** (10,581 imágenes)
   - Región: Tobillo/Pie

2. **FracAtlas** (~4,083 imágenes)
   - Regiones: Pierna, mano, cadera, hombro

**Total combinado**: ~14,664 imágenes

## ⚡ Opción 1: Entrenamiento Automático (RECOMENDADO)

Ejecuta un solo comando y todo se hace automáticamente:

```cmd
entrenar_completo.bat
```

Este script hará:
1. ✅ Preparar FracAtlas al formato correcto
2. ✅ Fusionar ambos datasets
3. ✅ Entrenar el modelo automáticamente

**Tiempo total**: ~40-50 minutos

## 🔧 Opción 2: Paso a Paso Manual

### Paso 1: Preparar FracAtlas

```cmd
py -3.12 scripts/preparar_fracatlas.py
```

Esto convertirá FracAtlas al formato `fractured` / `not fractured`.

### Paso 2: Fusionar Datasets

```cmd
py -3.12 scripts/merge_multiple_datasets_v2.py
```

Cuando pregunte "¿Continuar con la fusión?", escribe `s` y presiona Enter.

### Paso 3: Entrenar

```cmd
py -3.12 scripts/train_simple.py
```

El entrenamiento mostrará progreso en tiempo real.

## 📈 Durante el Entrenamiento

Verás algo como:

```
Época [1/50] - 82.1s
  Train Loss: 0.3644 | Train Acc: 0.8376
  Val Loss: 0.2127 | Val Acc: 0.9103
  Val Prec: 0.8978 | Val Rec: 0.9067
  ✓ Mejor modelo guardado! (Val Acc: 0.9103)
```

- **Cada época**: ~80-100 segundos
- **Early stopping**: Se detendrá si no mejora en 10 épocas
- **Progreso**: Cada 100 batches muestra avance

## ✅ Cuando Termine

1. El modelo se guardará en: `models/bone_classifier_v2/best_model.pth`

2. Actualiza la app web editando `app_web.py` línea 14:
   ```python
   MODEL_PATH = "models/bone_classifier_v2/best_model.pth"
   ```

3. Prueba el modelo:
   ```cmd
   py -3.12 test_modelo.py "ruta/a/imagen.jpg"
   ```

4. Inicia la app web:
   ```cmd
   py -3.12 app_web.py
   ```

## 🎯 Resultados Esperados

Con 2 datasets combinados (~14,664 imágenes):

- **Accuracy esperado**: 92-95%
- **Ventaja**: Funciona con múltiples regiones (pierna, mano, cadera, tobillo)
- **Tiempo de entrenamiento**: 30-40 minutos

## ⚠️ Troubleshooting

**Si el entrenamiento se cuelga:**
- Verifica que `train_simple.py` tenga `num_workers=0`
- El script ya está optimizado para Windows

**Si dice "Cargando datasets..." por mucho tiempo:**
- Es normal, está escaneando ~14,000 imágenes
- Espera 2-3 minutos

**Si falla por memoria:**
- El batch_size=32 debería funcionar con 8GB VRAM
- Si aún falla, edita `train_simple.py` y cambia `batch_size=32` a `batch_size=16`

## 📝 Resumen

```bash
# TODO EN UNO (más fácil):
entrenar_completo.bat

# O MANUAL:
py -3.12 scripts/preparar_fracatlas.py
py -3.12 scripts/merge_multiple_datasets_v2.py
py -3.12 scripts/train_simple.py
```

¡Listo! 🎉
