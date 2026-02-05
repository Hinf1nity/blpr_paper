from difflib import SequenceMatcher
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os

model_identification_plate = YOLO('ldp_best.pt')
model_ocr = YOLO('our_lpr_4.pt')


def image_rect(img):
    re_img = cv2.resize(img, (780, 540),
                        interpolation=cv2.INTER_LINEAR)
    copy_re_img = re_img.copy()
    gray = cv2.cvtColor(re_img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(
        gray, 150, 180, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    canny = cv2.Canny(thresh, 50, 100)
    contours = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                      reverse=True)[:10]

    points = []
    for c in contours:
        epsilon = 0.05 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) > 130000:
                cv2.drawContours(re_img, [approx], 0, (0, 255, 0), 3)
                for point in approx:
                    points.append((point[0][0], point[0][1]))
    if len(points) == 0:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                  np.ones((7, 7), np.uint8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                  np.ones((13, 13), np.uint8))
        canny = cv2.Canny(thresh, 50, 100)
        contours = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                          reverse=True)[:10]
        for c in contours:
            epsilon = 0.05 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                if cv2.contourArea(approx) > 130000:
                    cv2.drawContours(re_img, [approx], 0, (0, 255, 0), 3)
                    for point in approx:
                        points.append((point[0][0], point[0][1]))
    try:
        menor_y = min(points, key=lambda p: p[1])[0]
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        x_points.sort()
        y_points.sort()
        if menor_y <= x_points[0]+90:
            pts1 = np.float32([[x_points[1], y_points[0]], [x_points[3], y_points[1]],
                               [x_points[0], y_points[2]], [x_points[2], y_points[3]]])
        else:
            pts1 = np.float32([[x_points[0], y_points[1]], [x_points[2], y_points[0]],
                               [x_points[1], y_points[3]], [x_points[3], y_points[2]]])
        pts2 = np.float32([[0, 0], [x_points[3]-x_points[0], 0], [0, y_points[3] -
                          y_points[0]], [x_points[3]-x_points[0], y_points[3]-y_points[0]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(
            copy_re_img, M, (x_points[3]-x_points[0], y_points[3]-y_points[0]))
        return dst
    except Exception:
        return None


def Plates_Number(Model_Number):
    if model_ocr.names[Model_Number] in ["Bolivia", "_"]:
        return ''
    return (model_ocr.names[Model_Number])


class PlateEnhancer:
    def __init__(self):
        # Configuración de CLAHE para condiciones de baja luz
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def get_domain_id(self, image):
        """Clasifica el dominio basado en el brillo promedio."""
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0]
        avg_luma = np.mean(y_channel)
        print(f"Average Luma: {avg_luma}")
        if avg_luma < 90:
            return "LOW_LIGHT"
        elif avg_luma > 150:
            return "HIGH_LIGHT"
        return "NORMAL"

    def apply_gamma_correction(self, image, gamma=1.5):
        """Ajusta el deslumbramiento (Gamma > 1 oscurece, < 1 aclara)."""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance_by_domain(self, image):
        """Aplica preprocesamiento según la iluminación detectada."""
        domain = self.get_domain_id(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)

        if domain == "LOW_LIGHT":
            # Resaltar bordes y texto en la oscuridad
            enhanced = self.clahe.apply(gray)
            # Convertir de vuelta a BGR para que YOLO reciba 3 canales
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return cv2.erode(enhanced, kernel, iterations=1), domain

        elif domain == "HIGH_LIGHT" or domain == "NORMAL":
            # Reducir sobreexposición/reflejos
            enhanced = self.apply_gamma_correction(image, gamma=0.5)
            enhanced = cv2.dilate(enhanced, kernel, iterations=1)
            return enhanced, domain

    def order_points(self, pts):
        """Ordena 4 puntos: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process_plate(self, raw_crop):
        # 1. Mejorar imagen según iluminación
        enhanced_img, domain = self.enhance_by_domain(raw_crop)

        # # 2. Preparar para rectificación (tu lógica optimizada)
        # re_img = cv2.resize(enhanced_img, (780, 540),
        #                     interpolation=cv2.INTER_LINEAR)
        # gray = cv2.cvtColor(re_img, cv2.COLOR_BGR2GRAY)

        # # Usamos Adaptive Threshold porque es más robusto tras la mejora
        # _, thresh = cv2.threshold(
        #     gray, 150, 180, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
        #                           np.ones((7, 7), np.uint8))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
        #                           np.ones((13, 13), np.uint8))

        # canny = cv2.Canny(thresh, 50, 100)
        # contours, _ = cv2.findContours(
        #     canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # for c in contours:
        #     epsilon = 0.05 * cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, epsilon, True)

        #     if len(approx) == 4 and cv2.contourArea(approx) > 130000:
        #         pts1 = self.order_points(approx.reshape(4, 2))
        #         # Proporción estándar de placa (400x120)
        #         pts2 = np.float32([[0, 0], [400, 0], [400, 120], [0, 120]])

        #         M = cv2.getPerspectiveTransform(pts1, pts2)
        #         rectified = cv2.warpPerspective(re_img, M, (400, 120))
        #         rectified = cv2.resize(
        #             rectified, (700, 500), interpolation=cv2.INTER_AREA)
        #         return rectified, domain

        enhanced_img = cv2.resize(
            enhanced_img, (700, 500), interpolation=cv2.INTER_AREA)
        return enhanced_img, domain


def get_character_accuracy(real_text, pred_text):
    """Calcula la precisión basada en la similitud de caracteres."""
    if not real_text or not pred_text:
        return 0.0
    # Ratio de similitud (0.0 a 1.0)
    return SequenceMatcher(None, real_text, pred_text).ratio()


def char_by_char_metrics(real_text, pred_text):
    """
    Evalúa la precisión comparando la posición y contenido de los caracteres.
    Retorna: (Precisión %, caracteres_correctos, total_real)
    """
    real_text = str(real_text).strip().upper()
    pred_text = str(pred_text).strip().upper()

    # Caso de placa vacía
    if not real_text:
        return 0.0, 0, 0

    # Creamos una matriz para calcular la distancia de edición (Levenshtein)
    rows = len(real_text) + 1
    cols = len(pred_text) + 1
    dist = np.zeros((rows, cols), dtype=int)

    for i in range(1, rows):
        dist[i, 0] = i
    for i in range(1, cols):
        dist[0, i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if real_text[row-1] == pred_text[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row, col] = min(dist[row-1, col] + 1,      # Deletion
                                 dist[row, col-1] + 1,      # Insertion
                                 dist[row-1, col-1] + cost)  # Substitution

    # Caracteres correctos aproximados: N_total - Errores de edición
    # Usamos max(0, ...) para evitar negativos en casos de predicciones muy largas y erróneas
    ediciones = dist[rows-1, cols-1]
    correctos = max(0, len(real_text) - ediciones)

    accuracy = (correctos / len(real_text)) * 100
    return accuracy

# --- CONFIGURACIÓN ---


path_raiz = 'dataset_2_dividido_por_luz'
dominios = ['low_light', 'normal_light', 'high_light']
# Nombre del archivo CSV dentro de cada carpeta de dominio
CSV_NAME = 'placas_labels.csv'

enhancer = PlateEnhancer()
model_plate = YOLO('ldp_best.pt')
model_ocr = YOLO('our_lpr_4.pt')

# Diccionario para guardar métricas finales
stats = {d: [] for d in dominios}

for dominio in dominios:
    base_path = os.path.join(path_raiz, dominio)
    csv_path = os.path.join(base_path, CSV_NAME)
    img_dir = os.path.join(base_path)  # Ajustar a tu ruta

    if not os.path.exists(csv_path):
        print(f"No se encontró el CSV en {dominio}")
        continue

    # Cargar CSV: Se asume columnas ['filename', 'plate_text']
    df_gt = pd.read_csv(csv_path)

    print(f"\n>>>> ANALIZANDO PRECISIÓN EN: {dominio.upper()} <<<<")

    for index, row in df_gt.iterrows():
        img_name = row['imagen']
        real_plates = [row[f'plate{i}'] for i in range(1, 5)]
        # Eliminar NaN
        real_plates = [plate for plate in real_plates if pd.notna(plate)]

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        print(f"Img: {img_name}")

        # 1. Detectar Placa
        results_plate = model_plate.predict(
            source=img, save=False, verbose=False)

        for plate_result in results_plate:
            for plate_box in plate_result.boxes:
                if model_plate.names[int(plate_box.cls[0].item())] == "Placa":
                    x1, y1, x2, y2 = map(int, plate_box.xyxy[0].cpu().numpy())
                    crop = img[y1:y2, x1:x2]

                    # 2. Rectificación y Mejora
                    enhanced_img, _ = enhancer.process_plate(crop)
                    rectified_plate = image_rect(enhanced_img)
                    input_ocr = cv2.resize(
                        rectified_plate, (700, 500)) if rectified_plate is not None else enhanced_img

                    # 3. OCR de Caracteres
                    results_ocr = model_ocr.predict(
                        source=input_ocr, save=False, verbose=False)

                    detected_chars = []
                    for char_result in results_ocr:
                        for box in char_result.boxes:
                            if float(box.conf[0]) > 0.3:
                                x_char = box.xyxy[0][0].item()
                                char_label = Plates_Number(
                                    int(box.cls[0].item()))
                                detected_chars.append([x_char, char_label])

                    detected_chars.sort(key=lambda x: x[0])
                    pred_text = "".join([c[1]
                                        for c in detected_chars]).strip().upper()

                    # 4. Comparar con Ground Truth
                    real_plate = max(
                        real_plates, key=lambda rp: char_by_char_metrics(rp, pred_text))
                    acc = char_by_char_metrics(real_plate, pred_text)
                    stats[dominio].append(acc)
                    print(
                        f"Real: {real_plate} | Pred: {pred_text} | Acc: {acc:.2f}%")

# --- REPORTE FINAL ---
print("\n" + "="*30)
print("RESUMEN DE PRECISIÓN POR DOMINIO")
print("="*30)
for d in dominios:
    if stats[d]:
        avg_acc = np.mean(stats[d])
        print(f"{d.upper()}: {avg_acc:.2f}% de precisión media de caracteres")
    else:
        print(f"{d.upper()}: Sin datos")
