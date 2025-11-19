# 🦴 Bone Fracture Detection with AI

Sistema de detección de fracturas óseas usando Deep Learning con EfficientNetB3.

## 📋 Requisitos

- Python 3.12
- PyTorch con CUDA (RTX 4060 8GB)
- Gradio

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## 📊 Dataset

El proyecto usa el dataset **Bone_Fracture_Binary_Classification** con 10,581 imágenes de rayos X de tobillos/pies.

## 🎯 Uso

### Entrenar Modelo

```bash
py -3.12 scripts/train_simple.py
```

### Probar Modelo

```bash
py -3.12 test_modelo.py "ruta/a/imagen.jpg"
```

### Iniciar App Web

```bash
py -3.12 app_web.py
```

O usar el script:
```bash
iniciar_web.bat
```

## 📁 Estructura

```
proyecto-ia-main/
├── app_web.py              # Aplicación web Gradio
├── test_modelo.py          # Script de prueba
├── scripts/
│   ├── train_simple.py     # Entrenamiento simplificado
│   ├── train_bone_classifier.py  # Entrenamiento completo
│   ├── merge_multiple_datasets_v2.py  # Fusionar datasets
│   └── predict_fracture.py # Predicción por lotes
├── data/
│   └── Bone_Fracture_Binary_Classification/
└── models/
    └── bone_classifier/
        └── best_model.pth
```

## 🎯 Resultados

- **Accuracy**: 97.23%
- **Precision**: 96.69%
- **Recall**: 98.13%

## 📝 Notas

- El modelo actual está optimizado para fracturas de **tobillo/pie**
- Para otras regiones anatómicas, necesitas agregar datasets adicionales
- La app web corre en `http://localhost:7860`

## 🔗 Datasets Adicionales

Para mejorar la generalización del modelo:
- [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012)
- [Bone Fracture Multi-Region](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)

---
⚠️ Este proyecto es educativo y **no reemplaza** la valoración de un profesional de la salud.
