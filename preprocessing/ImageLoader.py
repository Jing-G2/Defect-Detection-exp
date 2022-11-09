import json
import os

import torch
from torch.utils.data.dataset import Dataset
import cv2
from torchvision import transforms


class ImageLoader(Dataset):
    def __init__(self, folder_path, transform=None, select_classes=None):
        if select_classes is None:
            select_classes = list(range(1, 21))
        self.image_path = folder_path
        images = sorted(os.listdir(self.image_path))
        self.images = [i for i in images if int(i.split('_')[0]) in select_classes]
        self.transform = self.__class__._transform
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_path, self.images[index])), cv2.COLOR_BGR2RGB)
        return self.transform(img), self.images[index]  # img, filename

    @staticmethod
    def _transform(x, y):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)
