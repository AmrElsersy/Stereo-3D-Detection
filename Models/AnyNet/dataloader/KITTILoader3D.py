import random
import torchvision.transforms as transforms

import numpy as np
from . import preprocess
import torch
import torch.utils.data as data
from PIL import Image
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return cv2.imread(path, 1)

def disparity_loader(path):
    pil_image =  Image.open(path)
    dataL = np.ascontiguousarray(pil_image, dtype=np.float32)/256
    return dataL

def npy_disparity_loader(path):
    return np.load(path).astype(np.float32)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, loader=default_loader, load_npy=False):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = disparity_loader
        if load_npy:
            self.dploader = npy_disparity_loader
        self.normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        left_img = torch.tensor(left_img, dtype=torch.float32).transpose(0, 1).T
        right_img = torch.tensor(right_img, dtype=torch.float32).transpose(0, 1).T

        c, h, w = left_img.shape

        left_img =  transforms.functional.crop(left_img, h - 352,  w - 1200, h, w)
        right_img =  transforms.functional.crop(right_img, h - 352,  w - 1200, h, w)

        left_img = transforms.Normalize(**self.normalize)(left_img)
        right_img = transforms.Normalize(**self.normalize)(right_img)

        dataL = dataL[h - 352:h, w - 1200:w]

        dataL = torch.from_numpy(dataL).float()
        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
