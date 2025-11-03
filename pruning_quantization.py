from ultralytics import YOLO
from torch.nn.utils import prune
from torch.quantization import quantize_dynamic

model = YOLO("best8_copy.pt", task='detect')
