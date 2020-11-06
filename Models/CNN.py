# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     CNN
   Description :
   Author :       walnut
   date:          2020/10/27
-------------------------------------------------
   Change Activity:
                  2020/10/27:
-------------------------------------------------
"""
__author__ = 'walnut'

import math
import torch
import torch.nn as nn
import torch.nn.functional as Func
from paras import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cov1 = nn.Sequential(
            nn.Conv2d(
                in_channels=7,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.cov2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.predict = nn.Linear(3840, NET_OUT_NUM)
        self.norm = torch.nn.BatchNorm1d(NET_OUT_NUM)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.5)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.cov1(x)
        x = self.cov2(x)
        x = x.view(x.size(0), -1)
        x = self.predict(x)
        return self.norm(x)
