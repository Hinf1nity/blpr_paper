import os
import cv2
import numpy as np
from ultralytics import YOLO

# ----------METRICS CALCULATION----------
DATA_PATH = "datasets/lpr_dataset/data.yaml"
SPLIT = "test"

dataset_root = os.path.dirname(os.path.abspath(DATA_PATH))
images_dir = os.path.join(dataset_root, SPLIT, "images")
labels_dir = os.path.join(dataset_root, SPLIT, "labels")

model = YOLO("runs/detect/train12/weights/best.pt")

metrics = model.val(data=DATA_PATH, split=SPLIT)
class_names = model.names

precision = metrics.box.p
recall = metrics.box.r
f1 = metrics.box.f1
ap50 = metrics.box.ap50
ap5095 = metrics.box.ap


def yolo_to_xyxy(box, img_w, img_h):
    xc, yc, w, h = box
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x1, y1, x2, y2]


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def compute_iou_per_class(model, images_dir, labels_dir, class_names):
    iou_dict = {i: [] for i in class_names.keys()}

    results = model.predict(images_dir, save=False, verbose=False)

    for r in results:
        img_h, img_w = r.orig_shape
        img_name = os.path.splitext(os.path.basename(r.path))[0]

        label_file = os.path.join(labels_dir, img_name + ".txt")
        if not os.path.exists(label_file):
            continue

        gts = []
        with open(label_file, "r") as f:
            for line in f:
                c, xc, yc, w, h = map(float, line.strip().split())
                box = yolo_to_xyxy([xc, yc, w, h], img_w, img_h)
                gts.append((int(c), box))

        preds = [(int(c), b.xyxy[0].cpu().numpy())
                 for b, c in zip(r.boxes, r.boxes.cls)]

        for pc, pb in preds:
            for gc, gb in gts:
                if pc == gc:
                    iou_dict[pc].append(iou(pb, gb))

    return {cls: np.mean(vals) if vals else 0 for cls, vals in iou_dict.items()}


dataset_root = os.path.dirname(os.path.abspath(DATA_PATH))
split_dir = os.path.join(dataset_root, SPLIT)
ious = compute_iou_per_class(model, images_dir, labels_dir, class_names)

print(f"{'Clase':<15}{'Precition':<12}{'Recall':<10}{'F1':<10}{'mAP0.5':<12}{'mAP0.5:0.95':<15}{'IoU':<10}")
print("="*85)

for i, name in class_names.items():
    print(
        f"{name:<15}{precision[i]:<12.3f}{recall[i]:<10.3f}{f1[i]:<10.3f}{ap50[i]:<12.3f}{ap5095[i]:<15.3f}{ious[i]:<10.3f}")
