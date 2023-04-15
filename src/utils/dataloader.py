#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/15 16:21
# @Author  : Wang Jixin
# @File    : dataloader.py
# @Software: PyCharm

from PIL import Image
from .transformers import data_transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .read_split_data import read_split_data


class MyDataset(Dataset):
    def __init__(self, root, images_path, images_label, transform):
        self.root = root
        self.images = images_path
        self.labels = images_label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)


BATCH_SIZE = 64

root = r"D:\研究生\研究生\machine_learning\src\raw_datasets\flower_photos"
train_images_path, train_images_label, test_images_path, test_images_label = read_split_data(root, 0.2)

train_data = MyDataset(root, train_images_path, train_images_label, transform=data_transform['train'])
test_data = MyDataset(root, test_images_path, test_images_label, transform=data_transform['test'])

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
