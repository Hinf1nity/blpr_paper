import os
from glob import glob

# Ruta raíz del primer dataset
labels_root = '/media/hinfinity/Disco Local/Diego/dataset_yolo/labels'

# Subcarpetas a procesar
splits = ['train', 'val', 'test']

for split in splits:
    label_dir = os.path.join(labels_root, split)
    label_files = glob(os.path.join(label_dir, '*.txt'))

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = '1'  # Cambiar clase 0 a clase 1
            new_lines.append(' '.join(parts) + '\n')

        # Sobrescribir el archivo
        with open(label_file, 'w') as f:
            f.writelines(new_lines)

print("✅ Todos los labels han sido actualizados de clase 0 → clase 1.")
