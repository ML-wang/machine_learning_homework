#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/15 15:11
# @Author  : Wang Jixin
# @File    : read_split_data.py
# @Software: PyCharm

import os
import json
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt


def read_split_data(root: str, test_rate: float = 0.2):
    """
    对图片数据集进行分割
    :param root: 数据集所在的路径(不同类型图片所在文件夹路径)
    :param test_rate: 验证集在数据集中所占的比例
    :return: 训练图片路径，训练图片标签，验证集图片路径，验证集图片标签
    """
    random.seed(0)  # 保证随机结果可复现

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = sorted([cla for cla in os.listdir(root)
                           if os.path.isdir(
            os.path.join(root, cla))])  # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # 排序，保证顺序一致
    # flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((test, key) for key, test in class_indices.items()), indent=4)
    with open('../output/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in flower_class:  # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

        cla_path = os.path.join(root,
                                cla)  # ['flower_data/daisy', 'flower_data/dandelion', 'flower_data/roses', 'flower_data/sunflowers', 'flower_data/tulips']
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        test_path = random.sample(images, k=int(len(images) * test_rate))

        for img_path in images:
            if img_path in test_path:  # 如果该路径在采样的验证集样本中则存入验证集
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training_dataset.".format(len(train_images_path)))
    print("{} images for test_dataset.".format(len(test_images_path)))

    # 绘制不同类别图片在 测试集 和 验证集 中的数量

    return train_images_path, train_images_label, test_images_path, test_images_label


def save_images_path(images_path: list, images_label: list, path):
    # images_labels = list(zip(images_path, '\t', images_label))
    images_labels = pd.DataFrame(list(zip(images_path, images_label)))
    images_labels.to_csv(f'../dataset/{path}', encoding='utf-8')


if __name__ == '__main__':
    root = r"D:\研究生\研究生\machine_learning\src\raw_datasets\flower_photos"
    train_images_path, train_images_label, test_images_path, test_images_label = read_split_data(root, 0.2)
    print(train_images_path,train_images_label)
    save_images_path(train_images_path, train_images_label, 'train_images_labels.csv')
    save_images_path(test_images_path, test_images_label, 'test_images_labels.csv')









    # print(test_images_path)

    # plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(flower_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(flower_class)), flower_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()

# def plot_data_loader_image(data_loader):
#     batch_size = data_loader.batch_size
#     plot_num = min(batch_size, 4)
# 
#     json_path = '/Users/steven/Documents/DeepLearning/DLShare/0707DatasetsDataLoader/LoadDataset/dataset/flower_photos/class_indices.json'
#     assert os.path.exists(json_path), json_path + " does not exist."
#     json_file = open(json_path, 'r')
#     class_indices = json.load(json_file)
# 
#     for data in data_loader:
#         images, labels = data
#         for i in range(plot_num):
#             # [C, H, W] -> [H, W, C]
#             img = images[i].numpy().transpose(1, 2, 0)
#             # 反Normalize操作
#             img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
#             label = labels[i].item()
#             plt.subplot(1, plot_num, i+1)
#             plt.xlabel(class_indices[str(label)])
#             plt.xticks([])  # 去掉x轴的刻度
#             plt.yticks([])  # 去掉y轴的刻度
#             plt.imshow(img.astype('uint8'))
#         plt.show()
