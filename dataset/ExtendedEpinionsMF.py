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
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.data_pth = data_pth
        dir_name, base_name = os.path.split(self.data_pth)
        self.mode = base_name
        self.dir_name = dir_name
        self.read_data()
        super(ExtendedEpinionsRateMF, self).__init__()

    def read_data(self):
        rate_pth = self.data_pth + '.rate'

        self.rate = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)

        f = open(os.path.join(self.dir_name, 'user2v.json'), 'r')
        self.user2item = json.load(f)['user2item']
        f.close()

    def __getitem__(self, idx):
        return self.rate[idx]

    def __len__(self):
        return len(self.rate)


class ExtendedEpinionsLinkMF(Dataset):
    def __init__(self, data_pth, config):
        self.config = config
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.data_pth = data_pth
        dir_name, base_name = os.path.split(self.data_pth)
        self.mode = base_name
        self.dir_name = dir_name
        self.read_data()
        super(ExtendedEpinionsLinkMF, self).__init__()

    def read_data(self):
        link_pth = self.data_pth + '.link'

        self.link = np.loadtxt(link_pth, delimiter=',', dtype=np.float32)

        f = open(os.path.join(self.dir_name, 'user2v.json'), 'r')
        self.user2trust = json.load(f)['user2trust']
        f.close()

    def __getitem__(self, idx):
        return self.link[idx]

    def __len__(self):
        return len(self.link)


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