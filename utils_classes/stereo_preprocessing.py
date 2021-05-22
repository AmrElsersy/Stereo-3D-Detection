from Models.AnyNet.dataloader import preprocess
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')

class StereoPreprocessing:
    def __init__(self, loader=default_loader):
        self.loader = loader

    def preprocess(self, left_img, right_img):
        # left_img = left_img.convert('RGB')
        # right_img = right_img.convert('RGB')
        left_img = Image.fromarray(np.uint8(left_img)).convert('RGB')
        right_img = Image.fromarray(np.uint8(right_img)).convert('RGB')
        w, h = left_img.size

        left_img = left_img.crop((w - 1200, h - 352, w, h))
        right_img = right_img.crop((w - 1200, h - 352, w, h))
        w1, h1 = left_img.size

        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img)
        right_img = processed(right_img)

        left_img = left_img.clone().detach().reshape(1, *left_img.size())
        right_img = right_img.clone().detach().reshape(1, *right_img.size())
        return left_img, right_img
