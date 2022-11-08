#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dataset.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/7 19:32   zxx      1.0         None
"""

# import lib
import dgl
import torch
from dgl.data import DGLDataset
import pandas as pd
import os
import numpy as np

class RateCommonDataset(DGLDataset):
    def __init__(self, config):
        self._g = None
        self.config = config
        self.pred_user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.model_name = config['MODEL']['model_name']
        self.task = config['TRAIN']['task']
        self.data_name = config['DATA']['data_name']
        self.data_pth = os.path.join('./data', self.data_name, 'splited_data')
        self.dir_pth = os.path.join(self.data_pth, self.model_name)

        print('=' * 20 + 'begin process' + '=' * 20)
        if self.has_cache() is False:
            self.process()
        else:
            self.load()
            print('=' * 20 + 'load graph finished' + '=' * 20)

        super(RateCommonDataset, self).__init__(name=self.data_name)

    def save(self):
        if not os.path.isdir(self.dir_pth):
            os.mkdir(self.dir_pth)
        graph_pth = os.path.join(self.dir_pth, f'{self.task}_graph.bin')
        dgl.save_graphs(graph_pth, self._g)
        info_pth = os.path.join(self.dir_pth, f'{self.task}_info.pkl')
        dgl.data.utils.save_info(info_pth, {'train_size': self.train_size, 'val_size': self.val_size})

    def load(self):
        graph_pth = os.path.join(self.dir_pth, f'{self.task}_graph.bin')
        self._g = dgl.load_graphs(graph_pth)[0][0]
        info_pth = os.path.join(self.dir_pth, f'{self.task}_info.pkl')
        size_dict = dgl.data.utils.load_info(info_pth)
        self.train_size = size_dict['train_size']
        self.val_size = size_dict['val_size']

    def process(self):
        record = {}
        u = np.empty(0)
        i = np.empty(0)
        # 将所有的边都读到总图中
        for mode in ['train', 'val', 'test']:
            rate_pth = os.path.join(self.data_pth, mode + '.rate')
            record[mode] = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)
            u = np.concatenate([u, record[mode][:, 0]])
            i = np.concatenate([i, record[mode][:, 1]])
        print('=' * 20 + 'read rate data finished' + '=' * 20)
        self.train_size = len(record['train'])
        self.val_size = len(record['val'])
        graph_data = {
            ('user', 'rate', 'item'): (u, i),
            ('item', 'rated-by', 'user'): (i, u)
        }
        num_nodes = {
            'user': self.pred_user_num,
            'item': self.item_num
        }
        self._g = dgl.heterograph(
            data_dict=graph_data,
            num_nodes_dict=num_nodes
        )
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

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return self._g


class SocialDataset(RateCommonDataset):
    def __init__(self, config, directed=False):
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.directed = directed
        super(SocialDataset, self).__init__(config)

    def save(self):
        if not os.path.isdir(self.dir_pth):
            os.mkdir(self.dir_pth)
        graph_pth = os.path.join(self.dir_pth, f'{self.task}_graph.bin')
        dgl.save_graphs(graph_pth, self._g)
        info_pth = os.path.join(self.dir_pth, f'{self.task}_info.pkl')
        dgl.data.utils.save_info(info_pth, {
            'train_size': self.train_size,
            'train_link_size': self.train_link_size,
            'val_size': self.val_size,
            'val_link_size': self.val_link_size
        })

    def load(self):
        graph_pth = os.path.join(self.dir_pth, f'{self.task}_graph.bin')
        self._g = dgl.load_graphs(graph_pth)[0][0]
        info_pth = os.path.join(self.dir_pth, f'{self.task}_info.pkl')
        size_dict = dgl.data.utils.load_info(info_pth)
        self.train_size = size_dict['train_size']
        self.train_link_size = size_dict['train_link_size']
        self.val_size = size_dict['val_size']
        self.val_link_size = size_dict['val_link_size']

    def process(self):
        record = {'rate': {}, 'link': {}}
        u = np.empty(0)
        i = np.empty(0)
        u1 = np.empty(0)
        v = np.empty(0)
        # 将所有的边都读到总图中
        for mode in ['train', 'val', 'test']:
            rate_pth = os.path.join(self.data_pth, mode + '.rate')
            record['rate'][mode] = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)
            u = np.concatenate([u, record['rate'][mode][:, 0]])
            i = np.concatenate([i, record['rate'][mode][:, 1]])

            link_pth = os.path.join(self.data_pth, mode + '.link')
            record['link'][mode] = np.loadtxt(link_pth, delimiter=',', dtype=np.float32)
            u1 = np.concatenate([u1, record['link'][mode][:, 0]])
            v = np.concatenate([v, record['link'][mode][:, 1]])

        print('=' * 20 + 'read rate data finished' + '=' * 20)
        self.train_size = len(record['rate']['train'])
        self.train_link_size = len(record['link']['train'])
        self.val_size = len(record['rate']['val'])
        self.val_link_size = len(record['link']['val'])

        if self.directed:
            graph_data = {
                ('user', 'rate', 'item'): (u, i),
                ('item', 'rated-by', 'user'): (i, u),
                ('user', 'trusted-by', 'user'): (v, u1)
            }
        else:
            graph_data = {
                ('user', 'rate', 'item'): (u, i),
                ('item', 'rated-by', 'user'): (i, u),
                ('user', 'trust', 'user'): (u1, v),
                ('user', 'trusted-by', 'user'): (v, u1)
            }

        num_nodes = {
            'user': self.total_user_num,
            'item': self.item_num
        }
        self._g = dgl.heterograph(
            data_dict=graph_data,
            num_nodes_dict=num_nodes
        )
        print('=' * 20 + 'construct graph finished' + '=' * 20)

        # 保存
        self.save()
        print('=' * 20 + 'save graph finished' + '=' * 20)