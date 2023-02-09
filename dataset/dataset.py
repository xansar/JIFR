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

class DGLRecDataset(DGLDataset):
    def __init__(self, config):
        self._g = None
        self.config = config
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.model_name = config['MODEL']['model_name']
        self.task = config['TRAIN']['task']
        self.data_name = config['DATA']['data_name']
        self.data_pth = os.path.join('./data', self.data_name, 'splited_data')
        self.bin_sep_lst = eval(config['METRIC'].get('bin_sep_lst', 'None'))
        self.dir_pth = os.path.join(self.data_pth, 'cache')
        self.is_log = config['LOG']
        self.print_info = print if self.is_log else lambda x: None
        if not os.path.isdir(self.dir_pth):
            os.makedirs(self.dir_pth)

        self.print_info('=' * 20 + 'begin process' + '=' * 20)
        if self.has_cache() is False:
            self.process()
        else:
            self.load()
            self.print_info('=' * 20 + 'load graph finished' + '=' * 20)

        # 根据rate和link的度分组
        if self.bin_sep_lst is not None:
            self.generate_bins_lst()

        super(DGLRecDataset, self).__init__(name=self.data_name)

    def save(self):
        if not os.path.isdir(self.dir_pth):
            os.mkdir(self.dir_pth)
        graph_pth = os.path.join(self.dir_pth, f'graph.bin')
        dgl.save_graphs(graph_pth, self._g)
        # info_pth = os.path.join(self.dir_pth, f'info.pkl')
        # info_dict = {
        #     'train_rate_size': self.train_rate_size,
        #     'val_rate_size': self.val_rate_size,
        #     'train_link_size': self.train_link_size,
        #     'val_link_size': self.val_link_size
        # }
        #
        # dgl.data.utils.save_info(info_pth, info_dict)

    def load(self):
        graph_pth = os.path.join(self.dir_pth, f'graph.bin')
        self._g = dgl.load_graphs(graph_pth)[0][0]
        # info_pth = os.path.join(self.dir_pth, f'info.pkl')
        # size_dict = dgl.data.utils.load_info(info_pth)

        # self.train_rate_size = size_dict['train_rate_size']
        # self.val_rate_size = size_dict['val_rate_size']
        # self.train_link_size = size_dict['train_link_size']
        # self.val_link_size = size_dict['val_link_size']

    def process(self):
        record = {'rate': {}, 'link': {}}
        u = np.empty(0)
        i = np.empty(0)
        r = np.empty(0)
        u1 = np.empty(0)
        v = np.empty(0)
        # 将所有的边都读到总图中
        for mode in ['train']:
            rate_pth = os.path.join(self.data_pth, mode + '.rate')
            record['rate'][mode] = np.loadtxt(rate_pth, delimiter=',', dtype=np.float32)
            u = np.concatenate([u, record['rate'][mode][:, 0]])
            i = np.concatenate([i, record['rate'][mode][:, 1]])
            r = np.concatenate([r, record['rate'][mode][:, 2]])

            link_pth = os.path.join(self.data_pth, mode + '.link')
            record['link'][mode] = np.loadtxt(link_pth, delimiter=',', dtype=np.float32)
            u1 = np.concatenate([u1, record['link'][mode][:, 0]])
            v = np.concatenate([v, record['link'][mode][:, 1]])

        self.print_info('=' * 20 + 'read rate data finished' + '=' * 20)
        # self.train_rate_size = len(record['rate']['train'])
        # self.train_link_size = len(record['link']['train'])
        # self.val_rate_size = len(record['rate']['val'])
        # self.val_link_size = len(record['link']['val'])

        graph_data = {
            ('user', 'rate', 'item'): (u, i),
            ('item', 'rated-by', 'user'): (i, u),
            ('user', 'trust', 'user'): (u1, v),
            ('user', 'trusted-by', 'user'): (v, u1)
        }

        num_nodes = {
            'user': self.user_num,
            'item': self.item_num
        }
        self._g = dgl.heterograph(
            data_dict=graph_data,
            num_nodes_dict=num_nodes
        )

        self._g.edges['rate'].data['rating'] = torch.tensor(r, dtype=torch.long)
        self._g.edges['rated-by'].data['rating'] = torch.tensor(r, dtype=torch.long)

        self.print_info('=' * 20 + 'construct graph finished' + '=' * 20)

        # 保存
        self.save()
        self.print_info('=' * 20 + 'save graph finished' + '=' * 20)

    def generate_bins_lst(self):
        # 分组起点终点，二维的，包含四个值，分别是rate起点、rate终点、link起点、link终点
        self.bins_start_end_lst = []
        self.u_lst = []
        self.bin_user_num = []
        self.rate_edges_num = []
        self.link_edges_num = []

        for i in range(len(self.bin_sep_lst)):
            # 获取对应的mask，找到train_g里面在当前度范围内的用户id
            rate_s = self.bin_sep_lst[i]
            self.bins_start_end_lst.append([])
            self.u_lst.append([])
            self.bin_user_num.append([])
            self.rate_edges_num.append([])
            self.link_edges_num.append([])
            for j in range(len(self.bin_sep_lst)):
                link_s = self.bin_sep_lst[j]
                if i == len(self.bin_sep_lst) - 1:
                    rate_e = '∞'
                    rate_mask = (self._g.out_degrees(etype='rate') >= rate_s)
                else:
                    rate_e = self.bin_sep_lst[i + 1]
                    rate_mask = (self._g.out_degrees(etype='rate') >= rate_s) * \
                           (self._g.out_degrees(etype='rate') < rate_e)

                if j == len(self.bin_sep_lst) - 1:
                    link_e = '∞'
                    link_mask = (self._g.out_degrees(etype='trust') >= link_s)
                else:
                    link_e = self.bin_sep_lst[j + 1]
                    link_mask = (self._g.out_degrees(etype='trust') >= link_s) * \
                           (self._g.out_degrees(etype='trust') < link_e)
                self.bins_start_end_lst[-1].append([rate_s, rate_e, link_s, link_e])
                mask = rate_mask * link_mask
                # 所有在这个度范围内的用户id tensor
                u_lst = self._g.nodes('user').masked_select(mask)
                # rate 边数
                rate_edges_u = self._g.edges(etype='rate')[0]
                # masked select配合isin 把所有在ulst中的u都选出来
                rate_e_id = torch.arange(rate_edges_u.shape[0]).masked_select(torch.isin(rate_edges_u, u_lst))
                rate_edges_num = rate_e_id.shape[0]
                # link 边数
                link_edges_u = self._g.edges(etype='trust')[0]
                # masked select配合isin 把所有在ulst中的u都选出来
                link_e_id = torch.arange(link_edges_u.shape[0]).masked_select(torch.isin(link_edges_u, u_lst))
                link_edges_num = link_e_id.shape[0]
                self.rate_edges_num[-1].append(rate_edges_num)
                self.link_edges_num[-1].append(link_edges_num)

                self.u_lst[-1].append(u_lst)
                self.bin_user_num[-1].append(len(u_lst))

        # 计算行列汇总
        tensor_bin_user_num = torch.tensor(self.bin_user_num)
        tensor_rate_edges_num = torch.tensor(self.rate_edges_num)
        tensor_link_edges_num = torch.tensor(self.link_edges_num)
        # 行汇总
        bin_user_num_row_sum = tensor_bin_user_num.sum(dim=1).tolist()
        rate_edges_num_row_sum = tensor_rate_edges_num.sum(dim=1).tolist()
        link_edges_num_row_sum = tensor_link_edges_num.sum(dim=1).tolist()
        for i in range(len(self.bin_sep_lst)):
            self.u_lst[i].append(torch.cat(self.u_lst[i]))
            self.bin_user_num[i].append(bin_user_num_row_sum[i])
            self.rate_edges_num[i].append(rate_edges_num_row_sum[i])
            self.link_edges_num[i].append(link_edges_num_row_sum[i])

        # 列汇总
        u_lst_col_sum = []
        for i in range(len(self.bin_sep_lst) + 1):
            cur_lst = []
            for j in range(len(self.bin_sep_lst)):
                cur_lst.append(self.u_lst[j][i]) # 这里注意次序，i表示列坐标，j表示行坐标，先j后i就是固定列然后滑动行
            u_lst_col_sum.append(torch.cat(cur_lst))
        assert len(u_lst_col_sum) == len(self.bin_sep_lst) + 1
        bin_user_num_col_sum = tensor_bin_user_num.sum(dim=0).tolist()
        rate_edges_num_col_sum = tensor_rate_edges_num.sum(dim=0).tolist()
        link_edges_num_col_sum = tensor_link_edges_num.sum(dim=0).tolist()

        self.u_lst.append(u_lst_col_sum)
        self.bin_user_num.append(bin_user_num_col_sum + [sum(bin_user_num_col_sum)])    # 前面是每列的和，后面是整个表格的和
        self.rate_edges_num.append(rate_edges_num_col_sum + [sum(rate_edges_num_col_sum)])
        self.link_edges_num.append(link_edges_num_col_sum + [sum(link_edges_num_col_sum)])
        assert len(self.u_lst) == len(self.bin_sep_lst) + 1 and len(self.u_lst[0]) == len(self.bin_sep_lst) + 1
        assert len(self.bin_user_num) == len(self.bin_sep_lst) + 1 and len(self.bin_user_num[0]) == len(self.bin_sep_lst) + 1
        assert len(self.rate_edges_num) == len(self.bin_sep_lst) + 1 and len(self.rate_edges_num[0]) == len(self.bin_sep_lst) + 1
        assert len(self.link_edges_num) == len(self.bin_sep_lst) + 1 and len(self.link_edges_num[0]) == len(self.bin_sep_lst) + 1

    def has_cache(self):
        is_graph = os.path.exists(os.path.join(self.dir_pth,
                                               f'graph.bin'))
        # is_info = os.path.exists(os.path.join(self.dir_pth,
        #                                       f'info.pkl'))
        is_info = True
        if is_info and is_graph:
            return True
        else:
            return False

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return self._g