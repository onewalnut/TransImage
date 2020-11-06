# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     MyDataset
   Description :
   Author :       walnut
   date:          2020/10/27
-------------------------------------------------
   Change Activity:
                  2020/10/27:
-------------------------------------------------
"""
__author__ = 'walnut'

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def default_transform(data):
    data = np.array(data)
    return torch.from_numpy(data.astype(np.int))


def image_transform(image):
    return transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])(image)


class TransImagesSet(Dataset):
    def __init__(self, trans_dicts, transformer=default_transform, image_transformer=image_transform):
        car_ids = []
        trans_images = []
        for car_id, trans_image in trans_dicts.items():
            car_ids.append(car_id)
            trans_images.append(trans_image)

        self.trans_images = trans_images
        self.labels = car_ids
        self.target_transform = transformer
        self.image_transform = image_transformer

    def __getitem__(self, idx):
        image = self.trans_images[idx]
        label = self.labels[idx]
        label_id = idx

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.target_transform is not None:
            label_id = self.target_transform(label_id)

        return image, label, label_id

    def __len__(self):
        return len(self.trans_images)
