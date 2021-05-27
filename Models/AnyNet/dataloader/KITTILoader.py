import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    pil_image =  Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def disparity_loader(path):
    return Image.open(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:  
            
            left_img = torch.tensor(left_img, dtype=torch.float32, device=device).transpose(0, 1).T
            right_img = torch.tensor(right_img, dtype=torch.float32, device=device).transpose(0, 1).T

            c, h, w = left_img.shape

            left_img =  transforms.functional.crop(left_img, h - 352,  w - 1200, h, w)
            right_img =  transforms.functional.crop(right_img, h - 352,  w - 1200, h, w)

            left_img = transforms.Normalize(**self.normalize)(left_img)
            right_img = transforms.Normalize(**self.normalize)(right_img)

            dataL = np.ascontiguousarray(dataL, dtype=np.float32)/256
            dataL = dataL[h - 352:h, w - 1200:w]
            dataL = torch.from_numpy(dataL).cuda()
            
        else:
            left_img = torch.tensor(left_img, dtype=torch.float32, device=device).transpose(0, 1).T
            right_img = torch.tensor(right_img, dtype=torch.float32, device=device).transpose(0, 1).T

            c, h, w = left_img.shape
        
            left_img =  transforms.functional.crop(left_img, h - 352,  w - 1200, h, w)
            right_img =  transforms.functional.crop(right_img, h - 352,  w - 1200, h, w)

            left_img = transforms.Normalize(**self.normalize)(left_img)
            right_img = transforms.Normalize(**self.normalize)(right_img)

            dataL = np.ascontiguousarray(dataL, dtype=np.float32)/256
            dataL = dataL[h - 352:h, w - 1200:w]
            dataL = torch.from_numpy(dataL).cuda()

        return left_img.cpu(), right_img.cpu(), dataL.cpu()

    def __len__(self):
        return len(self.left)
