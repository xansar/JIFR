#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py    
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

class EpinionsRateMF(Dataset):
    def __init__(self, data_pth, user_num=1597, item_num=24984):
        self.user_num = user_num
        self.item_num = item_num
        self.data_pth = data_pth
        self.read_data()
        super(EpinionsRateMF, self).__init__()

    def read_data(self):
        rate_pth = self.data_pth + '.rate'

        self.rate = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)

    def __getitem__(self, idx):
        return self.rate[idx]

    def __len__(self):
        return len(self.rate)

class EpinionsLinkMF(Dataset):
    def __init__(self, data_pth, user_num=1597, item_num=24984):
        self.user_num = user_num
        self.item_num = item_num
        self.data_pth = data_pth
        self.read_data()
        super(EpinionsLinkMF, self).__init__()

    def read_data(self):
        link_pth = self.data_pth + '.link'

        self.link = np.loadtxt(link_pth, delimiter=',', dtype=np.float32)

    def __getitem__(self, idx):
        return self.link[idx]

    def __len__(self):
        return len(self.link)

if __name__ == '__main__':
    rate_data = EpinionsRateMF(data_pth='../data/val')
    rate_loader = DataLoader(rate_data, batch_size=16, shuffle=False)
    for data in rate_loader:
        print(data)
        break

    link_data = EpinionsLinkMF(data_pth='../data/val')
    link_loader = DataLoader(link_data, batch_size=16, shuffle=False)
    for data in link_loader:
        print(data)
        break