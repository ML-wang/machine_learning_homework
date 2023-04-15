#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/15 14:51
# @Author  : Wang Jixin
# @File    : trainer.py
# @Software: PyCharm


import os
import yaml
import time
import random
import argparse
import warnings
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader


# 获得当前文件所在的绝对路径地址,D:\研究生\比赛\泰迪杯\teddy_cup_23\src
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# os.path.dirname() 去掉文件名,只返回上级目录  D:\研究生\比赛\泰迪杯\teddy_cup_23\src
# print(os.path.realpath(__file__))  # D:\研究生\比赛\泰迪杯\teddy_cup_23\src\pretrain.py

# 用作保存模型的名字
# time.strftime常用方法,将时间转化为格式化
start_time = time.strftime('%Y-%m-%d_%H-%M-%S_', time.localtime())
print(time.localtime())  # time.struct_time(tm_year=2023, tm_mon=4, tm_mday=10, tm_hour=15, tm_min=37, tm_sec=41, tm_wday=0, tm_yday=100, tm_isdst=0)
warnings.filterwarnings('ignore')


def train(model, data, optimizer, criterion, device,label):
    model.train()

    optimizer.zero_grad()
    pred = model(data, device)
    loss = criterion(pred,label)
    loss.backward()
    optimizer.step()
    return loss


def main(args):
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model
    model_config = yaml.load(open(os.path.join(CURRENT_PATH, 'config/model/' + model_name + '.yaml'), "r"),
                             Loader=yaml.FullLoader)

    seed = model_config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为特定cpu设置随机种子
    torch.cuda.manual_seed(seed) # 为特定gpu设置随机种子
    # torch.cuda.manual_seed_all(seed) # 使用多个为所有gpu设置随机种子


    data_path = args.data_path
    dataset = read_graph(data_path, col='post').to(device)
    dataset_config = {
        'x_feat_dim': len(dataset.x)
    }

    model = model_map[model_name](model_config, dataset_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')
    optimizer = optim.Adam(model.parameters(), lr=model_config['lr'], amsgrad=True, weight_decay=1e-12)
    criterion = NTXentLoss(device, len(dataset.x))

    best_loss = 1e8  # 设置训练停止误差
    print('Start training...')
    for epoch in range(model_config['epochs']):
        train_loss = train(model, dataset, optimizer, criterion, device)
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch,
                'best_loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(CURRENT_PATH, f'ckpt_pretrain/{start_time}.pt'))
        print('[%d/%d]\tTrain Loss:%.4f,\tBest Loss%.4f.' % (epoch + 1, model_config['epochs'], train_loss, best_loss))
    print('Finish training!')

#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_pretrain', action='store')
    parser.add_argument('--data_path', type=str, default='outputs/tables/post_keywords.csv', action='store')
    parser.add_argument('--comment', type=str, default='None', action='store')
    args, unknown = parser.parse_known_args()
    main(args)

