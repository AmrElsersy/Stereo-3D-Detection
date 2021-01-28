from AnyNet.dataloader import preprocess
from visualization.KittiDataset import *
from visualization.KittiUtils import *

class StereoPreprocessing:
    def __init__(self, training=False):
        self.training = training

    def preprocess(self, left_img, right_img):
        left_img = Image.fromarray(np.uint8(left_img)).convert('RGB')
        right_img = Image.fromarray(np.uint8(right_img)).convert('RGB')
        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img
        else:
            w, h = left_img.size

            left_img = left_img.crop((w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            # convert [3, 368, 1232] to tensor [1, 3, 368, 1232]
            left_img = left_img.clone().detach().reshape(1, *left_img.size())
            right_img = right_img.clone().detach().reshape(1, *right_img.size())

            return left_img, right_img