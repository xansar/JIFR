#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 20:34   zxx      1.0         None
"""

# import lib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

class ExtendedEpinionsRateMF(Dataset):
    def __init__(self, data_pth, config):
        self.config = config
        self.user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.model_name = config['MODEL']['model_name']
        self.data_pth = data_pth
        dir_name, base_name = os.path.split(self.data_pth)
        self.mode = base_name
        self.dir_name = dir_name
        self.read_data()
        super(ExtendedEpinionsRateMF, self).__init__()

    def read_data(self):
        rate_pth = self.data_pth + '.rate'

        self.data = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)
        self.normalize(5, 0)
        f = open(os.path.join(self.dir_name, 'user2v.json'), 'r')
        self.user2history = json.load(f)['user2item']
        f.close()

    def normalize(self, max=None, min=None):
        if max is None:
            max = np.max(self.data[:, 2])
        if min is None:
            min = np.min(self.data[:, 2])
        self.data[:, 2] = (self.data[:, 2] - min) / (max - min)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class ExtendedEpinionsLinkMF(ExtendedEpinionsRateMF):
    def __init__(self, data_pth, config):
        super(ExtendedEpinionsLinkMF, self).__init__(data_pth, config)

    def read_data(self):
        link_pth = self.data_pth + '.link'

        self.data = np.loadtxt(link_pth, delimiter=',', dtype=np.float32)
        self.normalize(1, 0)
        f = open(os.path.join(self.dir_name, 'user2v.json'), 'r')
        self.user2history = json.load(f)['user2trust']
        f.close()



if __name__ == '__main__':
    rate_data = ExtendedEpinionsRateMF(data_pth='../data/ExtendedEpinions/splited_data/MFModel/val')
    rate_loader = DataLoader(rate_data, batch_size=16, shuffle=False)
    for data in rate_loader:
        print(data)
        break

    link_data = ExtendedEpinionsLinkMF(data_pth='../data/ExtendedEpinions/splited_data/MFModel/val')
    link_loader = DataLoader(link_data, batch_size=16, shuffle=False)
    for data, neg in link_loader:
        print(data)
        break