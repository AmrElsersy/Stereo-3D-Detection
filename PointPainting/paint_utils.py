import numpy as np
import cv2
import torch
import torchvision.transforms.transforms as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def postprocessing(pred):
    pred = pred.argmax(dim=1).squeeze().detach().cpu().numpy()
    return pred

def preprocessing_kitti(image):
    if type(image) != np.array:
        image = np.asarray(image)

    new_shape = (1024,512)
    image = cv2.resize(image, new_shape)
    image = T.ToTensor()(image)

    image = image.unsqueeze(0).to(device)
    return image
