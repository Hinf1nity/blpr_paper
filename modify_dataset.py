import os
import yaml
import shutil


def cargar_nombres_desde_yaml(ruta_yaml):
    with open(ruta_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def obtener_ids_ordenados_yolov11(carpeta_labels, carpeta_destino='/home/hinfinity/Documents/datasets/dataset_lpr'):
    ruta_yaml = os.path.join(carpeta_labels, 'data.yaml')
    carpetas = [os.path.join(
        carpeta_labels, labels) for labels in ['train', 'valid', 'test']]
    carpetas_destino = [os.path.join(carpeta_destino, labels) for labels in [
        'train', 'valid', 'test']]
    nombres_clase = cargar_nombres_desde_yaml(ruta_yaml)

    for carpeta, carpeta_dest in zip(carpetas, carpetas_destino):
        carpeta_label = os.path.join(carpeta, 'labels')
        for archivo in os.listdir(carpeta_label):
            if not archivo.endswith('.txt'):
                continue

            ruta = os.path.join(carpeta_label, archivo)
            with open(ruta, 'r') as f:
                lineas = f.readlines()

            objetos = []
            for linea in lineas:
                partes = linea.strip().split()
                if len(partes) >= 5:
                    id_clase = partes[0]
                    x_center = float(partes[1])
                    nombre_clase = nombres_clase[int(id_clase)]
                    if nombre_clase not in ['Bolivia', '_']:
                        objetos.append(
                            (x_center, nombre_clase.upper()))

            # Ordenar por x_center (más a la izquierda primero)
            objetos.sort(key=lambda x: x[0])

            # Concatenar los IDs en un solo string
            ids_ordenados = [obj[1] for obj in objetos]
            # También puedes usar ' '.join(ids_ordenados)
            string_ids = ''.join(ids_ordenados)

            nuevo_nombre = os.path.splitext(archivo)[0] + '.jpg'
            nuevo_nombre = os.path.join(carpeta, 'images', nuevo_nombre)

            ruta_destino = os.path.join(carpeta_dest, string_ids + '.jpg')
            shutil.copy2(nuevo_nombre, ruta_destino)


carpeta = '/home/hinfinity/Documents/datasets/Bolivian_LPR-5'
ids_por_imagen = obtener_ids_ordenados_yolov11(carpeta)
