import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, limit=-1, split_file=None):
    left_fold = 'image_2/'
    right_fold = 'image_3/'

    if not split_file is None:
        with open(split_file) as f:
            test_files = ([(str(x.strip())+'.png') for x in f.readlines() if len(x) > 0])
    else:
        test_files = [x for x in os.listdir(os.path.join(filepath, 'training', right_fold))]

    if not limit == -1:
        test_files = test_files[:limit]

    left_val = [os.path.join(filepath, 'training', left_fold, img) for img in test_files]
    right_val = [os.path.join(filepath, 'training', right_fold, img) for img in test_files]
    return left_val, right_val
