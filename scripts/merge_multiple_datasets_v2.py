"""
Script para combinar múltiples datasets de fracturas con diferentes estructuras.
Soporta variaciones en nombres de clases (fractured/fracture, not fractured/normal).
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import shutil
from pathlib import Path
import random
from collections import defaultdict

# Mapeo de nombres de clases a estándar
CLASS_MAPPING = {
    'fractured': 'fractured',
    'fracture': 'fractured',
    'fracturas': 'fractured',
    'not fractured': 'not fractured',
    'normal': 'not fractured',
    'no fracture': 'not fractured',
    'not_fractured': 'not fractured',
}

def count_images(directory):
    """Cuenta imágenes en un directorio."""
    if not directory.exists():
        return 0
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        count += len(list(directory.glob(ext)))
    return count

def find_class_folders(dataset_path):
    """
    Encuentra carpetas de clases en un dataset.
    Soporta estructuras: train/val/test o directamente clases.
    """
    dataset_path = Path(dataset_path)
    found_classes = defaultdict(list)

    # Buscar en estructura train/val/test
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name.lower()
                    if class_name in CLASS_MAPPING:
                        found_classes[CLASS_MAPPING[class_name]].append(class_dir)

    # Si no hay estructura split, buscar directamente carpetas de clases
    if not found_classes:
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                if class_name in CLASS_MAPPING:
                    found_classes[CLASS_MAPPING[class_name]].append(class_dir)

    # También buscar en subdirectorio "Dataset" (caso especial)
    dataset_subdir = dataset_path / "Dataset"
    if dataset_subdir.exists() and not found_classes:
        for class_dir in dataset_subdir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                if class_name in CLASS_MAPPING:
                    found_classes[CLASS_MAPPING[class_name]].append(class_dir)

    return found_classes

def merge_datasets(dataset_configs, output_path, split_ratios=(0.8, 0.1, 0.1)):
    """
    Combina múltiples datasets en uno solo.

    Args:
        dataset_configs: Lista de tuplas (ruta, prefijo) para cada dataset
        output_path: Ruta donde guardar el dataset combinado
        split_ratios: (train, val, test) - deben sumar 1.0
    """

    output_path = Path(output_path)

    print("="*70)
    print("FUSIÓN DE DATASETS DE FRACTURAS (VERSIÓN AVANZADA)")
    print("="*70)
    print()

    # Validar ratios
    if abs(sum(split_ratios) - 1.0) > 0.001:
        raise ValueError("Los ratios deben sumar 1.0")

    train_ratio, val_ratio, test_ratio = split_ratios

    # Estructura para acumular todas las imágenes
    all_images = {
        'fractured': [],
        'not fractured': []
    }

    # Leer todos los datasets
    for dataset_path, prefix in dataset_configs:
        dataset_path = Path(dataset_path)

        print(f"\nProcesando dataset: {dataset_path}")
        print(f"Prefijo: {prefix}")
        print("-"*70)

        if not dataset_path.exists():
            print(f"⚠️  Dataset no encontrado, saltando...")
            continue

        # Encontrar carpetas de clases (flexible)
        class_folders = find_class_folders(dataset_path)

        if not class_folders:
            print(f"⚠️  No se encontraron carpetas de clases válidas")
            continue

        # Recopilar imágenes de todas las carpetas encontradas
        for standard_class, folders in class_folders.items():
            for class_dir in folders:
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    images = list(class_dir.glob(ext))
                    for img in images:
                        all_images[standard_class].append((img, prefix))

                count = count_images(class_dir)
                if count > 0:
                    print(f"  {class_dir.name} -> {standard_class}: {count} imágenes")

    print()
    print("="*70)
    print("RESUMEN DE IMÁGENES RECOPILADAS")
    print("="*70)

    total_fractured = len(all_images['fractured'])
    total_not_fractured = len(all_images['not fractured'])
    total = total_fractured + total_not_fractured

    print(f"\nFractured: {total_fractured} imágenes")
    print(f"Not Fractured: {total_not_fractured} imágenes")
    print(f"Total: {total} imágenes")

    if total == 0:
        print("\n⚠️  No se encontraron imágenes. Verifica las rutas de los datasets.")
        return

    # Análisis de balance
    if total_fractured > 0 and total_not_fractured > 0:
        ratio = max(total_fractured, total_not_fractured) / min(total_fractured, total_not_fractured)
        print(f"\nRatio de balance: {ratio:.2f}:1")
        if ratio > 3:
            print("⚠️  ADVERTENCIA: Las clases están muy desbalanceadas")
            print("   Considera técnicas de balanceo o pesos de clase en el entrenamiento")

    # Mezclar aleatoriamente
    print("\nMezclando imágenes aleatoriamente...")
    for class_name in all_images:
        random.shuffle(all_images[class_name])

    # Dividir en train/val/test
    print(f"\nDividiendo dataset ({train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test)...")

    splits_data = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }

    for class_name, images in all_images.items():
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits_data['train'][class_name] = images[:n_train]
        splits_data['val'][class_name] = images[n_train:n_train + n_val]
        splits_data['test'][class_name] = images[n_train + n_val:]

    # Copiar imágenes a la nueva estructura
    print("\nCopiando imágenes al dataset combinado...")

    total_copied = 0
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        for class_name in ['fractured', 'not fractured']:
            dst_dir = output_path / split / class_name

            copied = 0
            for img_path, prefix in splits_data[split][class_name]:
                dst_dir.mkdir(parents=True, exist_ok=True)
                new_name = f"{prefix}_{img_path.name}"
                dst_path = dst_dir / new_name

                if not dst_path.exists():
                    shutil.copy2(img_path, dst_path)
                    copied += 1

            total_copied += copied
            print(f"  {class_name}: {copied} imágenes copiadas")

    print()
    print("="*70)
    print("DATASET COMBINADO CREADO EXITOSAMENTE")
    print("="*70)
    print(f"\nRuta: {output_path}")
    print(f"Total de imágenes: {total_copied}")

    # Mostrar estadísticas finales
    print("\nEstadísticas finales:")
    for split in ['train', 'val', 'test']:
        fractured_count = count_images(output_path / split / 'fractured')
        not_fractured_count = count_images(output_path / split / 'not fractured')
        total_split = fractured_count + not_fractured_count

        print(f"\n{split.upper()}: {total_split} imágenes")
        print(f"  - Fractured: {fractured_count}")
        print(f"  - Not Fractured: {not_fractured_count}")

    print("\n✅ ¡Listo! Ahora puedes entrenar con:")
    print(f"   py -3.12 scripts/train_bone_classifier.py")
    print(f"\nEl script ya está configurado para usar:")
    print(f"   data_dir='{output_path}'")

if __name__ == "__main__":
    # CONFIGURACIÓN: Todos los datasets disponibles
    datasets = [
        ("data/Bone_Fracture_Binary_Classification", "original"),
        ("data/FracAtlas_Prepared", "fracatlas"),
    ]

    output = "data/Combined_Fracture_Dataset"

    print("\n📋 Datasets a combinar:")
    print()
    for path, prefix in datasets:
        dataset_path = Path(path)
        if dataset_path.exists():
            status = "✅"
            # Intentar contar imágenes
            class_folders = find_class_folders(dataset_path)
            total = sum(
                count_images(folder)
                for folders in class_folders.values()
                for folder in folders
            )
            print(f"  {status} {path}")
            print(f"      Prefijo: {prefix} | ~{total} imágenes")
        else:
            status = "❌"
            print(f"  {status} {path} (no encontrado)")

    print()
    print("="*70)
    response = input("¿Continuar con la fusión? (s/n): ")
    if response.lower() != 's':
        print("Operación cancelada.")
        exit(0)

    print()
    merge_datasets(
        dataset_configs=datasets,
        output_path=output,
        split_ratios=(0.8, 0.1, 0.1)  # 80% train, 10% val, 10% test
    )
