import yaml
import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('runs/retrain4/weights/ldp_best.pt')

with open("/home/hinfinity/Documents/datasets/Car_Plates_Bolivia_extended-3/data.yaml", "r") as f:
    data_config = yaml.safe_load(f)

val_images_path = '/home/hinfinity/Documents/datasets/Car_Plates_Bolivia_extended-3/test/images'
val_images = [os.path.join(val_images_path, f) for f in os.listdir(
    val_images_path) if f.endswith((".jpg", ".png", ".jpeg"))]

val_labels_path = val_images_path.replace("images", "labels")
val_labels = [img.replace("images", "labels").replace(
    ".jpg", ".txt").replace(".png", ".txt") for img in val_images]


def yolo2xy(yolo_box, img_width, img_height):
    x_center, y_center, w, h = yolo_box
    x1 = (x_center - w / 2) * img_width
    y1 = (y_center - h / 2) * img_height
    x2 = (x_center + w / 2) * img_width
    y2 = (y_center + h / 2) * img_height
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou


ious = []

for img_path, label_path in zip(val_images, val_labels):
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    gt_boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            values = list(map(float, line.strip().split()))
            gt_boxes.append(yolo2xy(values[1:], img_width, img_height))

    results = model(img_path)

    for result in results:
        pred_boxes = result.boxes.xyxy.cpu().numpy()

        for pred in pred_boxes:
            best_iou = 0
            for gt in gt_boxes:
                iou = compute_iou(pred, gt)
                best_iou = max(best_iou, iou)

            ious.append(best_iou)

mean_iou = np.mean(ious)
print(f"Mean IoU: {mean_iou:.6f}")
