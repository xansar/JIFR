#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ExtendedEpinionsMutualRec.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/30 15:56   zxx      1.0         None
"""

# import lib

import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd
import os
import json
import numpy as np

class ExtendedEpinionsJointMutualRec(DGLDataset):
    def __init__(self, config=None):
        self._g = None
        self.config = config
        self.pred_user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.model_name = config['MODEL']['model_name']
        self.task = config['TRAIN']['task']
        self.data_name = config['DATA']['data_name']
        # self.pred_user_num = 1598
        # self.item_num = 24985
        # self.total_user_num = 19790
        # self.model_name = 'MutualRec'
        # self.task = 'Joint'
        # self.data_name = 'ExtendedEpinions'
        self.data_pth = os.path.join('./data', self.data_name, 'splited_data')
        self.dir_pth = os.path.join(self.data_pth, self.model_name)

        print('=' * 20 + 'begin process' + '=' * 20)
        if self.has_cache() is False:
            self.process()
        else:
            self.load()
            print('=' * 20 + 'load graph finished' + '=' * 20)
        self.get_consumption_g_and_social_g()
        print('=' * 20 + 'get consumption and social subgraph' + '=' * 20)

        f = open(os.path.join(self.data_pth, 'user2v.json'), 'r')
        user2history = json.load(f)
        self.user2history = {
            'user2item': user2history['user2item'],
            'user2trust': user2history['user2trust'],
        }
        f.close()

        super(ExtendedEpinionsJointMutualRec, self).__init__(name=self.data_name)

    def save(self):
        if not os.path.isdir(self.dir_pth):
            os.mkdir(self.dir_pth)
        graph_pth = os.path.join(self.dir_pth, f'{self.task}_graph.bin')
        dgl.save_graphs(graph_pth, self._g)
        info_pth = os.path.join(self.dir_pth, f'{self.task}_info.pkl')
        dgl.data.utils.save_info(info_pth, {
            'train_size': self.train_size,
            'val_size': self.val_size,
            'test_size': self.test_size,
            'train_consumption_size': self.train_consumption_size,
            'val_consumption_size': self.val_consumption_size,
            'test_consumption_size': self.test_consumption_size,
        })

    def load(self):
        graph_pth = os.path.join(self.dir_pth, f'{self.task}_graph.bin')
        self._g = dgl.load_graphs(graph_pth)[0][0]
        info_pth = os.path.join(self.dir_pth, f'{self.task}_info.pkl')
        size_dict = dgl.data.utils.load_info(info_pth)

        self.train_size = size_dict['train_size']
        self.train_consumption_size = size_dict['train_consumption_size']
        self.val_size = size_dict['val_size']
        self.val_consumption_size = size_dict['val_consumption_size']
        self.test_size = size_dict['test_size']
        self.test_consumption_size = size_dict['test_consumption_size']

    def process(self):
        record = {
            'rate': {},
            'link': {}
        }
        u = np.empty(0)
        v = np.empty(0)
        # 将所有的边都读到总图中
        ## 总图构成：
        ## train_rate, train_link, val_rate, val_link, test_rate, test_link
        for mode in ['train', 'val', 'test']:
            rate_pth = os.path.join(self.data_pth, mode + '.rate')
            record['rate'][mode] = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)
            u = np.concatenate([u, record['rate'][mode][:, 0]])
            # 数据集里item从0开始计数，这里要放到所有user后面
            v = np.concatenate([v, record['rate'][mode][:, 1] + self.total_user_num])

            link_pth = os.path.join(self.data_pth, mode + '.link')
            record['link'][mode] = np.loadtxt(link_pth, delimiter=',', dtype=np.float32)
            u = np.concatenate([u, record['link'][mode][:, 0]])
            v = np.concatenate([v, record['link'][mode][:, 1]])
        print('=' * 20 + 'read rate and link data finished' + '=' * 20)

        self.train_size = len(record['rate']['train']) + len(record['link']['train'])
        self.train_consumption_size = len(record['rate']['train'])
        self.val_size = len(record['rate']['val']) + len(record['link']['val'])
        self.val_consumption_size = len(record['rate']['val'])
        self.test_size = len(record['rate']['test']) + len(record['link']['test'])
        self.test_consumption_size = len(record['rate']['test'])
        # 构建全图
        ## 节点数量为两类节点数量之和
        num_nodes = self.total_user_num + self.item_num
        self._g = dgl.graph((u, v), num_nodes=num_nodes)
        print('=' * 20 + 'construct graph finished' + '=' * 20)

        # 保存
        self.save()
        print('=' * 20 + 'save graph finished' + '=' * 20)

    def has_cache(self):
        is_graph = os.path.exists(os.path.join(self.dir_pth, f'{self.task}_graph.bin'))
        is_info = os.path.exists(os.path.join(self.dir_pth, f'{self.task}_info.pkl'))
        if is_info and is_graph:
            return True
        else:
            return False

    def get_consumption_g_and_social_g(self):
        consumption_eid = list(range(self.train_consumption_size)) + \
                          list(range(self.train_size, self.train_size + self.val_consumption_size)) + \
                          list(range(self.val_size, self.val_size + self.test_consumption_size))
        social_eid = list(range(self.train_consumption_size, self.train_size)) + \
                          list(range(self.val_consumption_size, self.test_size)) + \
                          list(range(self.test_consumption_size, self._g.num_edges()))
        self.consumption_g = dgl.edge_subgraph(self._g, consumption_eid, relabel_nodes=False)
        self.social_g = dgl.edge_subgraph(self._g, social_eid, relabel_nodes=False)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return self._g

if __name__ == '__main__':
    dataset = ExtendedEpinionsJointMutualRec()
    g = dataset[0]
    train_size = dataset.train_size
    train_g = dgl.edge_subgraph(g, range(train_size), relabel_nodes=False)
    print(train_g.edges())