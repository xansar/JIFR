#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ExtendedEpinionsLightGCN.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/25 20:18   zxx      1.0         None
"""

# import lib
import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd
import os
import numpy as np

class ExtendedEpinionsRateLightGCN(DGLDataset):
    def __init__(self, data_pth='../data/ExtendedEpinions/splited_data/', name='ExtendedEpinions'):
        self._g = None
        # self.data_pth = data_pth
        # self.train_mask = dict()
        # self.test_mask = dict()

        # self.config = config
        # self.user_num = eval(config['MODEL']['user_num'])
        # self.item_num = eval(config['MODEL']['item_num'])
        self.model_name = 'LightGCN'
        self.user_num = 1598
        self.item_num = 24985
        self.data_pth = data_pth
        self.dir_pth = os.path.join(self.data_pth, self.model_name)

        print('=' * 20 + 'begin process' + '=' * 20)
        if self.has_cache() is False:
            self.process()
        else:
            self.load()
            print('=' * 20 + 'load graph finished' + '=' * 20)

        super(ExtendedEpinionsRateLightGCN, self).__init__(name=name)

    def save(self):
        if not os.path.isdir(self.dir_pth):
            os.mkdir(self.dir_pth)
        graph_pth = os.path.join(self.dir_pth, 'graph.bin')
        dgl.save_graphs(graph_pth, self._g)
        info_pth = os.path.join(self.dir_pth, 'info.pkl')
        dgl.data.utils.save_info(info_pth, {'train_size': self.train_size, 'val_size': self.val_size})

    def load(self):
        graph_pth = os.path.join(self.dir_pth, 'graph.bin')
        self._g = dgl.load_graphs(graph_pth)[0][0]
        info_pth = os.path.join(self.dir_pth, 'info.pkl')
        size_dict = dgl.data.utils.load_info(info_pth)
        self.train_size = size_dict['train_size']
        self.val_size = size_dict['val_size']

    def process(self):
        record = {}
        u = np.empty(0)
        i = np.empty(0)
        # 将所有的边都读到总图中
        for mode in ['train', 'val', 'test']:
            rate_pth = self.data_pth + mode + '.rate'
            record[mode] = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)
            u = np.concatenate([u, record[mode][:, 0]])
            i = np.concatenate([i, record[mode][:, 1]])
        print('=' * 20 + 'read rate data finished' + '=' * 20)
        self.train_size = len(record['train'])
        self.val_size = len(record['val'])
        # 构建全图
        ## item的idx需要带一个偏置，使得item物品的序号从user之后开始计数
        i += self.user_num
        ## 节点数量为两类节点数量之和
        num_nodes = self.user_num + self.item_num
        self._g = dgl.graph((u, i), num_nodes=num_nodes)
        print('=' * 20 + 'construct graph finished' + '=' * 20)

        # 保存
        self.save()
        print('=' * 20 + 'save graph finished' + '=' * 20)

    def has_cache(self):
        is_graph = os.path.exists(os.path.join(self.dir_pth, 'graph.bin'))
        is_info = os.path.exists(os.path.join(self.dir_pth, 'info.pkl'))
        if is_info and is_graph:
            return True
        else:
            return False

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return self._g


if __name__ == '__main__':
    dataset = ExtendedEpinionsRateLightGCN()
    g = dataset[0]
    train_size = dataset.train_size
    train_g = dgl.edge_subgraph(g, range(train_size))
    print(train_g.edges())