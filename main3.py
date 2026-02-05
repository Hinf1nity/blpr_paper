import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict
import ollama
import base64
import torch
import multiprocessing as mp
from functools import lru_cache
import hashlib


final_project = YOLO("runs/retrain4/weights/ldp_best.pt", task='detect')
char_model = YOLO("our_lpr_4.pt", task='detect')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cap = cv2.VideoCapture('IMG_1265.MOV')


tracked_objects_past = {}
plate_history = defaultdict(list)
plate_confidence = defaultdict(float)
frame_count = 0
OCR_INTERVAL = 2  # frames
MIN_DETECTIONS = 3  # minimo de detecciones
LOW_CONFIDENCE_THRESHOLD = 0.5  # umbral para usar LLM
PLATE_CACHE = {}


class LLMPlateReader:
    def __init__(self, model_name="llama3.2-vision"):
        self.model_name = model_name
        self.setup_ollama_optimizations()

    def setup_ollama_optimizations(self):

        self.ollama_options = {
            'temperature': 0.1,
            'num_predict': 50,
            'top_k': 10,
            'top_p': 0.9,
            'repeat_penalty': 1.1,
            'num_ctx': 512,
            'num_batch': 8,
            'num_gpu': 1 if torch.cuda.is_available() else 0,
            'num_thread': mp.cpu_count() // 2,
            'use_mmap': True,
            'use_mlock': True
        }

    def image_to_base64(self, image):

        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def get_image_hash(self, image):

        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        return hashlib.md5(image_bytes).hexdigest()

    def cached_plate_inference(self, image, prompt="Analyze this license plate image and return ONLY the alphanumeric characters you can see on the plate. Return only the plate number without any additional text or explanation."):
        image_hash = self.get_image_hash(image)
        if image_hash and image_hash in PLATE_CACHE:
            return PLATE_CACHE[image_hash]

        base64_image = self.image_to_base64(image)
        if not base64_image:
            return ""

        messages = [{
            'role': 'user',
            'content': prompt,
            'images': [base64_image]
        }]

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options=self.ollama_options
        )

        result = response['message']['content'].strip()
        clean_result = ''.join(c for c in result if c.isalnum()).upper()
        if image_hash and len(clean_result) >= 6:
            PLATE_CACHE[image_hash] = clean_result
            if len(PLATE_CACHE) > 20:
                oldest_key = next(iter(PLATE_CACHE))
                del PLATE_CACHE[oldest_key]

        print(f"LLM resultadoo: {clean_result}")
        return clean_result


llm_reader = LLMPlateReader()
print("LLM iniciado correctamente :)")


def Plates_Number(Model_Number):
    if char_model.names[Model_Number] in ["Bolivia", "_"]:
        return ''
    return (char_model.names[Model_Number])


def get_domain_id(image):
    """Clasifica el dominio basado en el brillo promedio."""
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    y_channel = yuv[:, :, 0]
    avg_luma = np.mean(y_channel)
    print(f"Average Luma: {avg_luma}")
    if avg_luma < 90:
        return "LOW_LIGHT"
    elif avg_luma > 150:
        return "HIGH_LIGHT"
    return "NORMAL"


def apply_gamma_correction(image, gamma=1.5):
    """Ajusta el deslumbramiento (Gamma > 1 oscurece, < 1 aclara)."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def enhance_by_domain(image):
    """Aplica preprocesamiento según la iluminación detectada."""
    domain = get_domain_id(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)

    if domain == "LOW_LIGHT":
        # Resaltar bordes y texto en la oscuridad
        enhanced = clahe.apply(gray)
        # Convertir de vuelta a BGR para que YOLO reciba 3 canales
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return cv2.erode(enhanced, kernel, iterations=1), domain

    elif domain == "HIGH_LIGHT" or domain == "NORMAL":
        # Reducir sobreexposición/reflejos
        enhanced = apply_gamma_correction(image, gamma=0.5)
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)
        return enhanced, domain


def image_rect(img):
    img, _ = enhance_by_domain(img)
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


def preprocess_plate(roi):
    resized_Image = cv2.resize(roi, (700, 500), interpolation=cv2.INTER_AREA)

    return resized_Image


def extract_rois(results, target_label="Placa"):
    rois = []
    coordinates = []
    for result in results:
        for box in result.boxes:
            if final_project.names[int(box.cls)] == target_label:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = result.orig_img[max(y1-10, 0):y2+10, max(x1-10, 0):x2+10]
                rois.append(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                coordinates.append([x1-10, y1-10, x2+10, y2+10])
    return rois, coordinates


def display_results(frame, results):
    for result in results:
        for box in result.boxes:
            if final_project.names[int(box.cls)] == "Auto":
                car_box = list(map(int, box.xyxy[0]))
                color = (0, 255, 0)
                text = "Car"
                x1, y1, x2, y2 = car_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return frame


def get_most_frequent_plate(plate_id):
    if plate_id not in plate_history or len(plate_history[plate_id]) < MIN_DETECTIONS:
        return ""
    plate_counts = {}
    for plate_text in plate_history[plate_id]:
        if len(plate_text) >= 6:
            plate_counts[plate_text] = plate_counts.get(plate_text, 0) + 1

    if not plate_counts:
        return ""
    return max(plate_counts, key=plate_counts.get)


def should_process_ocr(plate_id):

    if plate_id not in tracked_objects_past:
        return True

    if len(plate_history[plate_id]) < MIN_DETECTIONS:
        return True

    if frame_count % OCR_INTERVAL == 0:
        return True

    return False


def annotate_plates(frame, plates, coords, new_ids_bool):
    global tracked_objects_past, plate_history, plate_confidence
    keys = list(tracked_objects_past.keys())
    for roi, coord, plate_id in zip(plates, coords, keys):
        x1, y1, x2, y2 = coord
        if should_process_ocr(plate_id):
            rect_roi = image_rect(roi)
            if rect_roi is not None:
                binarized_plate = preprocess_plate(rect_roi)
                original_plate = rect_roi
            else:
                roi_cropped = roi[10:roi.shape[0]-10, 10:roi.shape[1]-10]
                binarized_plate = preprocess_plate(roi_cropped)
                original_plate = roi_cropped

            # cv2.imshow('Binarized Plate', binarized_plate)

            char_results = char_model(binarized_plate)
            detected_chars = []

            for char_result in char_results:
                for box in char_result.boxes:
                    coord_x = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    Detected_Character = Plates_Number(cls)

                    if Detected_Character != '' and confidence > 0.3:
                        detected_chars.append(
                            [coord_x[0], Detected_Character, confidence])

            detected_chars.sort(key=lambda x: x[0])
            print(f"Detected characters for ID {plate_id}: {detected_chars}")

            plate_text = ""
            avg_confidence = 0
            use_llm = False

            if len(detected_chars) >= 6:
                avg_confidence = np.mean([char[2] for char in detected_chars])

                if avg_confidence > 0.6:
                    plate_text = "".join([char[1] for char in detected_chars])
                    print(
                        f"OCR alta confianza ({avg_confidence:.2f}): {plate_text}")

                elif avg_confidence > LOW_CONFIDENCE_THRESHOLD:
                    plate_text = "".join([char[1] for char in detected_chars])
                    print(
                        f"OCR confianza media ({avg_confidence:.2f}): {plate_text}")

                else:
                    use_llm = True
                    print(
                        f"OCR baja confianza ({avg_confidence:.2f}) - usando LLM")

            else:
                use_llm = True
                print(
                    f"Pocos caracteres detectados ({len(detected_chars)}) - usando LLM")

            if use_llm and llm_reader is not None:
                try:

                    llm_image = cv2.cvtColor(original_plate, cv2.COLOR_RGB2BGR)
                    llm_result = llm_reader.cached_plate_inference(llm_image)

                    if llm_result and len(llm_result) >= 6:
                        plate_text = llm_result
                        avg_confidence = 0.8
                        print(f"LLM resultado exitoso: {plate_text}")
                    else:
                        if detected_chars:
                            plate_text = "".join([char[1]
                                                 for char in detected_chars])
                            print(f"LLM falló, usando OCR: {plate_text}")
                        else:
                            plate_text = ""

                except Exception as e:
                    print(f"Error con LLM: {e}")
                    if detected_chars:
                        plate_text = "".join([char[1]
                                             for char in detected_chars])
                        avg_confidence = np.mean(
                            [char[2] for char in detected_chars])

            if plate_text and len(plate_text) >= 6:
                if plate_id not in plate_history:
                    plate_history[plate_id] = []
                plate_history[plate_id].append(plate_text)

                if len(plate_history[plate_id]) > 10:
                    plate_history[plate_id] = plate_history[plate_id][-10:]
                best_plate = get_most_frequent_plate(plate_id)
                if best_plate:
                    tracked_objects_past[plate_id] = best_plate
                    plate_confidence[plate_id] = avg_confidence

        if tracked_objects_past.get(plate_id, '') != '':
            text = tracked_objects_past[plate_id]
            conf_text = f" ({plate_confidence.get(plate_id, 0):.2f})"
            display_text = text + conf_text
        else:
            display_text = "Processing..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
    *'mp4v'), fps, (frame_width, frame_height))

# tracked_objects_past = {}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = final_project.track(
        frame, persist=True, tracker='bytetrack.yaml', conf=0.5, iou=0.3, max_det=15)

    # frame = display_results(frame, results)
    plate_rois, plate_coords = extract_rois(results, target_label="Placa")

    placas_results = results[0].boxes.cls.cpu().tolist()
    placas_results = [final_project.names[int(
        cls)] == "Placa" for cls in placas_results]

    if results[0].boxes.id is None:
        print("No objects detected :(")
        tracked_objects = {}
    else:
        tracked_objects = results[0].boxes.id.cpu().numpy().tolist()
        tracked_objects = {obj: tracked_objects_past.get(obj, "") for obj, flag in zip(
            tracked_objects, placas_results) if flag}

    print(f"Tracked objects: {tracked_objects}")

    # llimpiar historiales
    current_ids = set(tracked_objects.keys())
    ids_to_remove = set(plate_history.keys()) - current_ids
    for old_id in ids_to_remove:
        if old_id in plate_history:
            del plate_history[old_id]
        if old_id in plate_confidence:
            del plate_confidence[old_id]
        if old_id in tracked_objects_past:
            del tracked_objects_past[old_id]

    new_ids_bool = (tracked_objects.keys() == tracked_objects_past.keys())
    tracked_objects_past.update(tracked_objects)

    if plate_rois:
        annotate_plates(frame, plate_rois, plate_coords, new_ids_bool)

    out.write(frame)
    # cv2.imshow('frame', frame)

    # if cv2.waitKey(1) == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
