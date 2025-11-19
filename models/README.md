# Modelos Entrenados

Esta carpeta contiene los modelos entrenados.

## Estructura

- **bone_classifier/** - Modelos de clasificación binaria de fracturas
  - `best_model.pth` - Mejor modelo entrenado
  - `training_history.png` - Gráficos de entrenamiento
  - `confusion_matrix.png` - Matriz de confusión
  - `history.npy` - Historial de entrenamiento

- **cnn/** - Modelos YOLO para detección de objetos (legacy)

## Uso

Los modelos se generan automáticamente al ejecutar:

```bash
python scripts/train_bone_classifier.py
```

El modelo se guardará en `models/bone_classifier/best_model.pth`
