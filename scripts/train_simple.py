"""
Script de entrenamiento SIMPLIFICADO - Sin complicaciones
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}", flush=True)

class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        print(f"\n📂 Cargando: {root_dir}", flush=True)

        # Buscar imágenes fractured
        frac_dir = Path(root_dir) / 'fractured'
        if frac_dir.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img in frac_dir.glob(ext):
                    self.samples.append((str(img), 0))
            print(f"  ✓ Fractured: {sum(1 for _, l in self.samples if l == 0)}", flush=True)

        # Buscar imágenes not fractured
        nofrac_dir = Path(root_dir) / 'not fractured'
        if nofrac_dir.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img in nofrac_dir.glob(ext):
                    self.samples.append((str(img), 1))
            print(f"  ✓ Not Fractured: {sum(1 for _, l in self.samples if l == 1)}", flush=True)

        print(f"  📊 Total: {len(self.samples)} imágenes\n", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            # Imagen corrupta, devolver imagen negra
            image = Image.new('RGB', (224, 224))
            if self.transform:
                image = self.transform(image)
            return image, label

def train():
    print("="*70, flush=True)
    print("ENTRENAMIENTO SIMPLIFICADO - 3 DATASETS", flush=True)
    print("="*70, flush=True)

    # Transformaciones simples
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Cargar datasets
    print("\n🔄 CARGANDO DATASETS...\n", flush=True)
    train_dataset = SimpleDataset('data/Combined_Fracture_Dataset/train', train_transform)
    val_dataset = SimpleDataset('data/Combined_Fracture_Dataset/val', val_transform)
    test_dataset = SimpleDataset('data/Combined_Fracture_Dataset/test', val_transform)

    # DataLoaders
    print("🔄 Creando DataLoaders...\n", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Modelo
    print("🔄 Creando modelo EfficientNetB3...\n", flush=True)
    model = models.efficientnet_b3(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, 2)
    )
    model = model.to(device)

    print("✓ Modelo cargado en GPU\n", flush=True)

    # Optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Entrenamiento
    output_dir = Path('models/bone_classifier_v2')
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0
    patience_counter = 0

    print("="*70, flush=True)
    print("INICIANDO ENTRENAMIENTO", flush=True)
    print("="*70, flush=True)
    print(flush=True)

    for epoch in range(50):
        start_time = time.time()

        # TRAIN
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Progreso
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}", flush=True)

        train_acc = accuracy_score(train_labels, train_preds)
        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, average='binary')
        val_rec = recall_score(val_labels, val_preds, average='binary')
        val_loss /= len(val_loader)

        epoch_time = time.time() - start_time

        # Mostrar resultados
        print(f"\nÉpoca [{epoch+1}/50] - {epoch_time:.1f}s", flush=True)
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}", flush=True)
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", flush=True)
        print(f"  Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}", flush=True)

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Mejor modelo guardado! (Val Acc: {val_acc:.4f})", flush=True)
            patience_counter = 0
        else:
            patience_counter += 1

        print(flush=True)

        # Early stopping
        if patience_counter >= 10:
            print(f"\n⚠️  Early stopping en época {epoch+1}", flush=True)
            break

    # TEST
    print("="*70, flush=True)
    print("EVALUACIÓN FINAL EN TEST SET", flush=True)
    print("="*70, flush=True)

    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='binary')
    test_rec = recall_score(test_labels, test_preds, average='binary')
    test_f1 = f1_score(test_labels, test_preds, average='binary')

    print(f"\n📊 RESULTADOS FINALES:", flush=True)
    print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)", flush=True)
    print(f"  Test Precision: {test_prec:.4f}", flush=True)
    print(f"  Test Recall:    {test_rec:.4f}", flush=True)
    print(f"  Test F1-Score:  {test_f1:.4f}", flush=True)
    print(flush=True)

    print("="*70, flush=True)
    print("✅ ENTRENAMIENTO COMPLETADO", flush=True)
    print("="*70, flush=True)
    print(f"\nModelo guardado en: {output_dir / 'best_model.pth'}", flush=True)

if __name__ == "__main__":
    train()
