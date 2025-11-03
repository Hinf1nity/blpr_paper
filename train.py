import torch
from ultralytics import YOLO
import os
import random
from shutil import copyfile
import pickle

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

if __name__ == '__main__':
    my_model = YOLO('yolov8n.pt')
    my_model.train(data='data.yaml', epochs=60, batch=4, amp=True)

    my_model.export()