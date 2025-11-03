from torch.utils.data import Dataset
import os
import cv2
import random
import numpy as np
from imutils import paths
from PIL import Image

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
IDX2CHAR = {i: char for i, char in enumerate(CHARS)}


class LPRCLIPDataset(Dataset):
    def __init__(self, img_dirs, imgSize, PreprocFun=None, transform_clip=None):
        self.img_paths = []
        for dir in img_dirs:
            self.img_paths += [el for el in paths.list_images(dir)]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.PreprocFun = PreprocFun
        self.transform_clip = transform_clip

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform_clip:
            img = self.transform_clip(img)

        # Extraer texto desde el nombre
        basename = os.path.basename(img_path)
        name = basename.split("-")[0].split("_")[0]
        texto = ''.join(name)

        return img, texto
