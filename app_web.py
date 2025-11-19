"""
Aplicación web para detectar fracturas óseas usando el modelo entrenado.
Interfaz simple con Gradio.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/bone_classifier_v2/best_model.pth"

def create_model(num_classes=2):
    """Crea la arquitectura del modelo."""
    model = models.efficientnet_b3(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model

def load_model():
    """Carga el modelo entrenado."""
    model = create_model(num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

# Cargar modelo
print("Cargando modelo...")
model = load_model()
print("Modelo cargado exitosamente!")

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    """Predice si hay fractura en la imagen."""
    if image is None:
        return "Por favor, sube una imagen de rayos X"

    try:
        # Convertir a PIL si es necesario
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Convertir a RGB
        image = image.convert('RGB')

        # Transformar
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predicción
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        # Resultados (orden alfabético: fractured=0, not fractured=1)
        classes = ['Fractured (Fracturado)', 'Not Fractured (No Fracturado)']
        result = classes[predicted.item()]
        conf_percent = confidence.item() * 100

        # Probabilidades individuales (fractured=índice 0, not fractured=índice 1)
        fractured_prob = probs[0][0].item() * 100
        not_fractured_prob = probs[0][1].item() * 100

        # Crear mensaje detallado
        if predicted.item() == 0:  # fractured
            emoji = "⚠️"
            color = "red"
            message = f"# {emoji} FRACTURA DETECTADA\n\n"
        else:  # not fractured (índice 1)
            emoji = "✅"
            color = "green"
            message = f"# {emoji} NO SE DETECTÓ FRACTURA\n\n"

        message += f"**Confianza:** {conf_percent:.2f}%\n\n"
        message += f"### Probabilidades:\n"
        message += f"- 🔴 Fracturado: {fractured_prob:.2f}%\n"
        message += f"- 🟢 No Fracturado: {not_fractured_prob:.2f}%\n\n"

        if conf_percent > 95:
            message += "**Alta confianza** en el diagnóstico\n\n"
        elif conf_percent > 85:
            message += "**Confianza moderada-alta** en el diagnóstico\n\n"
        elif conf_percent > 70:
            message += "**Confianza moderada** - Considere una segunda opinión\n\n"
        else:
            message += "⚠️ **ADVERTENCIA: Baja confianza (<70%)**\n\n"
            message += "La imagen podría ser muy diferente al dataset de entrenamiento.\n"
            message += "Este modelo fue entrenado para: tobillo, pie, pierna, mano, cadera, hombro.\n"
            message += "No se recomienda usar esta predicción.\n\n"

        # Información técnica
        message += f"---\n"
        message += f"**Detalles técnicos:**\n"
        message += f"- Clase predicha: {predicted.item()} ({'fractured' if predicted.item() == 0 else 'not fractured'})\n"
        message += f"- Logits: [{outputs[0][0].item():.4f}, {outputs[0][1].item():.4f}]\n"
        message += f"- Tamaño de imagen procesada: 224x224px\n"

        return message

    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

# Crear interfaz
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Sube una imagen de rayos X"),
    outputs=gr.Markdown(label="Resultado del Diagnóstico"),
    title="🦴 Detector de Fracturas Óseas con IA",
    description="""
    ## Sistema de Detección de Fracturas con Deep Learning

    **Precisión del modelo: ~98%**

    ### Instrucciones:
    1. Sube una imagen de rayos X (formato JPG, PNG, etc.)
    2. El modelo analizará la imagen automáticamente
    3. Recibirás el diagnóstico con el nivel de confianza

    ### Información del modelo:
    - Arquitectura: EfficientNetB3
    - Dataset: ~14,664 imágenes de rayos X (2 datasets combinados)
    - Regiones: Tobillo, pie, pierna, mano, cadera, hombro
    - Entrenado en: RTX 4060
    - Validation Accuracy: ~98%

    ⚠️ **Nota importante:** Este sistema es una herramienta de apoyo diagnóstico.
    Siempre consulte con un profesional médico calificado.
    """,
    examples=[
        ["data/Bone_Fracture_Binary_Classification/test/fractured/0.png"],
        ["data/Bone_Fracture_Binary_Classification/test/not fractured/3.png"],
    ] if False else None,  # Cambia a True si quieres ejemplos
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    print("="*70)
    print("INICIANDO APLICACIÓN WEB - DETECTOR DE FRACTURAS")
    print("="*70)
    print(f"\nModelo: {MODEL_PATH}")
    print(f"Dispositivo: {device}")
    print(f"Dataset: ~14,664 imágenes (2 datasets)")
    print(f"Regiones: Tobillo, pie, pierna, mano, cadera, hombro")
    print(f"Precisión: ~98%")
    print("\nLa aplicación se abrirá en tu navegador...")
    print("="*70)

    demo.launch(
        server_name="0.0.0.0",  # Permite acceso desde otras máquinas
        server_port=7860,
        share=False,  # Cambia a True para obtener link público
        show_error=True
    )
