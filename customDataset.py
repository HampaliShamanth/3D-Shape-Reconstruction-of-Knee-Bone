import torch
import os
from skimage import io
from torch.utils.data import (Dataset)
import numpy as np


class EdgeImages(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):

        index=index+1 #There is no 0.png file
        img_path = os.path.join(self.root_dir, '%d.png' % (index))
        image = io.imread(img_path)

        target = image[:, 1024:] / 10 - 1000
        target = np.matrix.flatten(target)
        target = target[target != -1000]

        # The first 3957 elements belongs to the x coordinates of femur, tibia and patella
        target_X_Range = np.ptp(target[0:3957])
        scaleFactor = 100 / target_X_Range
        target = torch.tensor(target * scaleFactor)

        # Divide by 1000 because it was multipled in matlab
        # Multiply by 255 so that it can be converted to uint8 so that, that could be converted to tensor uint8
        image = (image[:, :1024]/1000)*255
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, target
