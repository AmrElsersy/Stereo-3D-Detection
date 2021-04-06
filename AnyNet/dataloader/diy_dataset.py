import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, log, split_file):
    left_fold = 'left/'
    right_fold = 'right/'
    disp_L = 'velodyne/'

    all_index = np.arange(4)
    trainist = all_index[:3]
    train = ['{:06d}.png'.format(x) for x in trainist]

    # train = [x for x in os.listdir(os.path.join(filepath, 'training', left_fold)) if is_image_file(x)]
    left_train = [os.path.join(filepath,  left_fold, img) for img in train]
    right_train = [os.path.join(filepath,  right_fold, img) for img in train]
    # left_train_disp = [os.path.join(filepath,  disp_L, img) for img in train]
    left_train_disp = [os.path.join(filepath,  right_fold, img) for img in train]
    
    # np.random.shuffle(all_index)
    all_index = np.arange(4)
    vallist = all_index[:4]

    
    val = ['{:06d}.png'.format(x) for x in vallist]
    # val = [x for x in os.listdir(os.path.join(filepath, 'validation', left_fold)) if is_image_file(x)]
    left_val = [os.path.join(filepath, left_fold, img) for img in val]
    right_val = [os.path.join(filepath, right_fold, img) for img in val]
    # left_val_disp = [os.path.join(filepath, disp_L, img) for img in val]
    left_val_disp = [os.path.join(filepath, right_fold, img) for img in val]

    return  left_train, right_train, left_train_disp, left_val, right_val, left_val_disp

def testloader(filepath):
    left_fold = 'left/'
    right_fold = 'right/'

    all_index = np.arange(4)
    # np.random.shuffle(all_index)
    vallist = all_index[:1]

    val = ['{:06d}.png'.format(x) for x in vallist]
    # val = [x for x in os.listdir(os.path.join(filepath, 'validation', left_fold)) if is_image_file(x)]
    left_val = [os.path.join(filepath, left_fold, img) for img in val]
    right_val = [os.path.join(filepath, right_fold, img) for img in val]
    left_val_disp = [os.path.join(filepath, right_fold, img) for img in val]

    return left_val, right_val, left_val_disp

