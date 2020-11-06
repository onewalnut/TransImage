# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     Vector2Image
   Description :
   Author :       walnut
   date:          2020/10/27
-------------------------------------------------
   Change Activity:
                  2020/10/27:
-------------------------------------------------
"""
__author__ = 'walnut'


import numpy as np
from PIL import Image


def pixel_value_norm(seq):
    # channels = []
    # for idx in range(seq.shape[1]):
    #     channel = seq[:, idx]
    #     channel = np.array(((channel - np.min(channel)) / (np.max(channel) - np.min(channel))) * 255 + 0.5, dtype=int)
    #     channels.append(channel)
    # return np.transpose(channels)

    seq = np.array(((np.array(seq) - np.min(seq)) / (np.max(seq) - np.min(seq))) * 255 + 0.5, dtype=int)
    return seq


def vector_to_image(seq, image_size=(32, 32)):

    seq_extend = pixel_value_norm(seq)

    idx = 0
    image = Image.new("RGB", image_size)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            image.putpixel((i, j), tuple(seq_extend[idx]))
            idx = idx + 1
    return image
