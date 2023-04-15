#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/15 15:08
# @Author  : Wang Jixin
# @File    : transformers.py
# @Software: PyCharm

from torchvision import transforms


data_transform = {
    "train": transforms.Compose([transforms.RandomCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 期望，标准差
    "test": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
