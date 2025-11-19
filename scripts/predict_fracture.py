"""
Script de inferencia para predecir fracturas óseas en nuevas imágenes.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(num_classes=2):
    """Crea la arquitectura del modelo (debe coincidir con el entrenamiento)."""
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


def load_model(model_path):
    """Carga el modelo entrenado."""
    model = create_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, img_size=224):
    """Predice si hay fractura en una imagen."""

    # Transformaciones (deben coincidir con las de validación)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cargar y transformar imagen
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    classes = ['Fractured', 'Not Fractured']
    result = {
        'prediction': classes[predicted.item()],
        'confidence': confidence.item() * 100,
        'fractured_prob': probs[0][0].item() * 100,
        'not_fractured_prob': probs[0][1].item() * 100
    }

    return result


def main():
    if len(sys.argv) < 2:
        print("Uso: python predict_fracture.py <ruta_imagen>")
        print("Ejemplo: python predict_fracture.py data/test/image.png")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "models/bone_classifier/best_model.pth"

    if not Path(image_path).exists():
        print(f"Error: No se encuentra la imagen {image_path}")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Error: No se encuentra el modelo {model_path}")
        print("Primero entrena el modelo con: python scripts/train_bone_classifier.py")
        sys.exit(1)

    print(f"Cargando modelo desde {model_path}...")
    model = load_model(model_path)

    print(f"Analizando imagen: {image_path}")
    result = predict_image(model, image_path)

    print("\n" + "="*60)
    print("RESULTADO DE LA PREDICCIÓN")
    print("="*60)
    print(f"Predicción: {result['prediction']}")
    print(f"Confianza: {result['confidence']:.2f}%")
    print(f"\nProbabilidades:")
    print(f"  - Fractured: {result['fractured_prob']:.2f}%")
    print(f"  - Not Fractured: {result['not_fractured_prob']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
