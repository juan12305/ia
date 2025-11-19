"""
Script de prueba rápida del modelo.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/bone_classifier/best_model.pth"

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

def test_model(image_path):
    """Prueba el modelo con una imagen."""

    # Cargar modelo
    print(f"Cargando modelo desde {MODEL_PATH}...")
    model = create_model(num_classes=2)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Modelo cargado. Métricas guardadas:")
    print(f"  Val Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"  Val Precision: {checkpoint['val_precision']:.4f}")
    print(f"  Val Recall: {checkpoint['val_recall']:.4f}")
    print()

    # Cargar imagen
    print(f"Cargando imagen: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Tamaño original: {image.size}")

    # Transformar
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Tensor shape: {image_tensor.shape}")
    print()

    # Predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    print("="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"\nLogits: {outputs[0]}")
    print(f"\nProbabilidades después de softmax:")
    print(f"  Clase 0 (fractured): {probs[0][0].item():.6f} ({probs[0][0].item()*100:.2f}%)")
    print(f"  Clase 1 (not fractured): {probs[0][1].item():.6f} ({probs[0][1].item()*100:.2f}%)")

    predicted = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted].item()

    print(f"\nPredicción: Clase {predicted}")
    print(f"Nombre de clase: {'fractured' if predicted == 0 else 'not fractured'}")
    print(f"Confianza: {confidence*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_modelo.py <ruta_imagen>")
        print("\nEjemplos:")
        print("  python test_modelo.py data/Bone_Fracture_Binary_Classification/test/fractured/0.png")
        print("  python test_modelo.py data/Bone_Fracture_Binary_Classification/test/not\\ fractured/3.png")
        sys.exit(1)

    test_model(sys.argv[1])
