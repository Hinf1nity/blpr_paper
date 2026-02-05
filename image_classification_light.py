import csv
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def classify_dataset_by_light(source_path, output_path, thresholds=(90, 150)):
    """
    Divide un dataset de YOLOv8 en 3 niveles de iluminación.
    thresholds: (límite_bajo, límite_alto) para el canal Y (0-255)
    """
    # Definir subdominios
    domains = ['low_light', 'normal_light', 'high_light']
    splits = ['train', 'valid', 'test']

    # Crear estructura de carpetas de destino
    for domain in domains:
        # for split in splits:
        os.makedirs(os.path.join(output_path, domain,
                                 ), exist_ok=True)
        # os.makedirs(os.path.join(output_path, domain,
        #             split, 'labels'), exist_ok=True)

    # Extensiones de imagen soportadas
    img_formats = ('.jpg', '.jpeg', '.png', '.bmp')

    # for split in splits:
    img_dir = os.path.join(source_path)
    # lbl_dir = os.path.join(source_path, split, 'labels')

    # if not os.path.exists(img_dir):
    #     print(f"Saltando {split}: No se encontró la carpeta de imágenes.")
    #     continue

    print(f"Procesando split: ...")
    files = [f for f in os.listdir(
        img_dir) if f.lower().endswith(img_formats)]

    for img_name in tqdm(files):
        # 1. Cargar imagen y calcular brillo en YUV
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convertir a YUV y extraer canal Y (Luma)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        avg_luma = np.mean(yuv[:, :, 0])

        # 2. Determinar categoría
        if avg_luma < thresholds[0]:
            category = 'low_light'
        elif avg_luma > thresholds[1]:
            category = 'high_light'
        else:
            category = 'normal_light'

        # 3. Definir rutas de destino
        target_img_path = os.path.join(
            output_path, category, img_name)

        # Buscar etiqueta correspondiente (.txt)
        # label_name = os.path.splitext(img_name)[0] + '.txt'
        # src_label_path = os.path.join(lbl_dir, label_name)
        # target_label_path = os.path.join(
        #     output_path, category, split, 'labels', label_name)

        # 4. Copiar archivos (usamos copy para no destruir el original)
        shutil.copy2(img_path, target_img_path)
        # if os.path.exists(src_label_path):
        #     shutil.copy2(src_label_path, target_label_path)


# --- CONFIGURACIÓN ---
# Cambia 'dataset_roboflow' por el nombre de tu carpeta descargada
# SOURCE_DATASET = 'dataset_2'
# OUTPUT_FOLDER = 'dataset_2_dividido_por_luz'

# classify_dataset_by_light(SOURCE_DATASET, OUTPUT_FOLDER)
# print("¡Proceso terminado!")


def generate_simple_csv(folder_path, output_csv):
    # Extensiones de imagen comunes
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # Obtener lista de archivos filtrada
    files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(extensions)]

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['filename'])  # Encabezado

        for f in files:
            writer.writerow([f])

    print(
        f"CSV generado exitosamente: {output_csv} con {len(files)} imágenes.")


generate_simple_csv(
    'dataset_2_dividido_por_luz/normal_light2', 'lista_imagenes_normal.csv')
