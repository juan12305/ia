"""
Script para preparar FracAtlas al formato de clasificación binaria.
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import shutil
from pathlib import Path

def preparar_fracatlas():
    """Convierte FracAtlas al formato fractured/not fractured."""

    print("="*70, flush=True)
    print("PREPARANDO DATASET FRACATLAS", flush=True)
    print("="*70, flush=True)
    print(flush=True)

    # Leer CSV
    csv_path = Path("data/FracAtlas/dataset.csv")
    if not csv_path.exists():
        print("❌ No se encontró dataset.csv", flush=True)
        return

    df = pd.read_csv(csv_path)
    print(f"Total de imágenes en CSV: {len(df)}", flush=True)
    print(flush=True)

    # Crear estructura de salida
    output_dir = Path("data/FracAtlas_Prepared")

    # Contar fracturas
    fractured_count = df['fractured'].sum()
    not_fractured_count = len(df) - fractured_count

    print(f"Fracturas: {fractured_count}", flush=True)
    print(f"No fracturas: {not_fractured_count}", flush=True)
    print(flush=True)

    # Crear directorios
    for split in ['train', 'val', 'test']:
        for cls in ['fractured', 'not fractured']:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    # Dividir datos (80% train, 10% val, 10% test)
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['fractured'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['fractured'])

    print("Copiando imágenes...", flush=True)

    copied = 0
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for _, row in split_df.iterrows():
            img_name = row['image_id']
            is_fractured = row['fractured']

            # Ruta fuente
            src = Path("data/FracAtlas/images") / img_name

            if not src.exists():
                continue

            # Ruta destino
            cls_name = 'fractured' if is_fractured == 1 else 'not fractured'
            dst = output_dir / split_name / cls_name / img_name

            # Copiar
            shutil.copy2(src, dst)
            copied += 1

            if copied % 500 == 0:
                print(f"  Copiadas {copied} imágenes...", flush=True)

    print(f"\n✅ Total copiado: {copied} imágenes", flush=True)
    print(flush=True)

    # Estadísticas finales
    print("Estadísticas finales:", flush=True)
    for split in ['train', 'val', 'test']:
        frac_count = len(list((output_dir / split / 'fractured').glob('*')))
        no_frac_count = len(list((output_dir / split / 'not fractured').glob('*')))
        print(f"\n{split.upper()}:", flush=True)
        print(f"  Fractured: {frac_count}", flush=True)
        print(f"  Not Fractured: {no_frac_count}", flush=True)

    print(flush=True)
    print("="*70, flush=True)
    print("LISTO! Dataset preparado en: data/FracAtlas_Prepared", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    preparar_fracatlas()
