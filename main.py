import numpy as np
import cv2
from ultralytics import YOLO
import easyocr

final_project = YOLO("runs/retrain4/weights/ldp_best.pt", task='detect')
char_model = YOLO(
    "yolo_modify/runs/detect/train3/weights/best.pt", task='detect')
# reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture('tests/traffic_test.mp4')


def Plates_Number(Model_Number):
    if char_model.names[Model_Number] in ["Bolivia", "_"]:
        return ''
    return (char_model.names[Model_Number])


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


def annotate_plates(frame, plates, coords, new_ids_bool):
    keys = list(tracked_objects_past.keys())
    placas = list(tracked_objects_past.values())
    blank_items = [placa == '' for placa in placas]
    new_ids_bool = not new_ids_bool
    for roi, coord, placas_keys in zip(plates, coords, keys):
        x1, y1, x2, y2 = coord
        if new_ids_bool or any(blank_items):
            rect_roi = image_rect(roi)
            if rect_roi is not None:
                binarized_plate = preprocess_plate(rect_roi)
            else:
                roi = roi[10:roi.shape[0]-10, 10:roi.shape[1]-10]
                binarized_plate = preprocess_plate(roi)
            cv2.imshow('Binarized Plate', binarized_plate)
            char_results = char_model(binarized_plate)
            detected_chars = []
            for char_result in char_results:
                for box in char_result.boxes:
                    # Assuming the class for characters is correctly labeled
                    coord_x = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].item())
                    Detected_Character = Plates_Number(cls)
                    if Detected_Character != '':
                        detected_chars.append([coord_x[0], Detected_Character])

            # Join and print the detected characters
            detected_chars.sort(key=lambda x: x[0])
            print(f"Detected characters: {detected_chars}")
            if len(detected_chars) <= 5:
                print("Not enough characters detected")
                text = "Can't extract text"
            else:
                text = "".join([char[1] for char in detected_chars])
                tracked_objects_past[placas_keys] = text

        else:
            if tracked_objects_past[placas_keys] != '':
                text = tracked_objects_past[placas_keys]
            else:
                text = "Can't extract text"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
    *'mp4v'), fps, (frame_width, frame_height))
tracked_objects_past = {}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = final_project.track(
        frame, persist=True, tracker='bytetrack.yaml', conf=0.5)
    frame = display_results(frame, results)
    plate_rois, plate_coords = extract_rois(results, target_label="Placa")
    placas_results = results[0].boxes.cls.cpu().tolist()
    placas_results = [final_project.names[int(
        cls)] == "Placa" for cls in placas_results]
    if results[0].boxes.id is None:
        print("No objects detected.")
        tracked_objects = {}
    else:
        tracked_objects = results[0].boxes.id.cpu().numpy().tolist()
        tracked_objects = {obj: "" for obj, flag in zip(
            tracked_objects, placas_results) if flag}
    print(f"Tracked objects: {tracked_objects}")
    new_ids_bool = (tracked_objects.keys() == tracked_objects_past.keys())
    if not new_ids_bool:
        tracked_objects_past = tracked_objects
    annotate_plates(
        frame, plate_rois, plate_coords, new_ids_bool)
    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
