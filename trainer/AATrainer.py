#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AATrainer.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/4 21:19   zxx      1.0         None
"""

# import lib

import dgl
import networkx as nx
import torch
import numpy as np
from tqdm import tqdm, trange
import os

from typing import List

from .BaseTrainer import BaseTrainer

class AATrainer(BaseTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            loss_func: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler,
            metric,
            dataset,
            config,
    ):
        super(AATrainer, self).__init__(config)
        # 读取数据
        self.dataset = dataset
        self.g = dataset[0]
        self.train_size = dataset.train_size
        self.val_size = dataset.val_size
        self.train_link_size = dataset.train_link_size
        self.val_link_size = dataset.val_link_size

        # 读取训练有关配置
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.neg_num = eval(config['DATA']['neg_num'])
        self.train_neg_num = eval(config['DATA']['train_neg_num'])
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])

        # 读取metric配置
        self.metric = metric
        self.ks = eval(config['METRIC']['ks'])

        # 其他配置
        self.device = config['TRAIN']['device']
        self._to(self.device)

    def get_graphs(self):
        train_edges = {
            # ('user', 'rate', 'item'): range(self.train_rate_size),
            # ('item', 'rated-by', 'user'): range(self.train_rate_size),
            ('user', 'trust', 'user'): range(self.train_link_size),
            ('user', 'trusted-by', 'user'): range(self.train_link_size)
        }
        train_g = dgl.edge_subgraph(self.g, train_edges, relabel_nodes=False)

        val_edges = {
            # ('user', 'rate', 'item'): range(self.train_rate_size, self.train_rate_size + self.val_size),
            # ('item', 'rated-by', 'user'): range(self.train_rate_size, self.train_rate_size + self.val_size),
            ('user', 'trust', 'user'): range(self.train_link_size, self.train_link_size + self.val_link_size),
            ('user', 'trusted-by', 'user'): range(self.train_link_size, self.train_link_size + self.val_link_size)
        }
        val_pred_g = dgl.edge_subgraph(self.g, val_edges, relabel_nodes=False)

        test_edges = {
            # ('user', 'rate', 'item'): range(self.train_rate_size + self.val_size, self.g.num_edges(('user', 'rate', 'item'))),
            # ('item', 'rated-by', 'user'): range(self.train_rate_size + self.val_size, self.g.num_edges(('item', 'rated-by', 'user'))),
            ('user', 'trust', 'user'): range(self.train_link_size + self.val_link_size, self.g.num_edges(('user', 'trusted-by', 'user'))),
            ('user', 'trusted-by', 'user'): range(self.train_link_size + self.val_link_size, self.g.num_edges(('user', 'trusted-by', 'user')))
        }
        test_pred_g = dgl.edge_subgraph(self.g, test_edges, relabel_nodes=False)
        # val_g = dgl.edge_subgraph(self.g, range(self.train_rate_size + self.val_size), relabel_nodes=False)

        return train_g, val_pred_g, test_pred_g

    def get_edges_lst(self, graph, etype='trust'):
        neg_u, neg_v = graph.edges(etype=etype)
        edges_lst = torch.cat([neg_u.unsqueeze(1), neg_v.unsqueeze(1)],
                                          dim=1).tolist()
        return edges_lst
    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            pass
        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                message_g = inputs['message_g']
                pred_g = inputs['pred_g']
                neg_g = self.construct_negative_graph(pred_g, self.neg_num,
                                                      etype=('user', 'trust', 'user'))
                pred_g_edges_lst = self.get_edges_lst(pred_g)
                neg_g_edges_lst = self.get_edges_lst(neg_g)

                pos_pred, neg_pred = self.model(
                    message_g,
                    pred_g_edges_lst,
                    neg_g_edges_lst
                )
                neg_pred = neg_pred.reshape(-1, self.neg_num)

                if self.bin_sep_lst is not None and self.is_visulized == True:
                    self.log_pred_histgram(pos_pred, neg_pred, mode)
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                return None
        else:
            raise ValueError("Wrong Mode")

    def get_side_info(self):
        return dgl.edge_type_subgraph(self.train_g, [('user', 'trust', 'user'), ('user', 'trusted-by', 'user')])

    def train(self):
        loss_name = ['Loss']
        self._train(loss_name)

    def _train(self, loss_name, side_info: dict=None):
        tqdm.write(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))

        self.train_g, self.val_pred_g, self.test_pred_g = self.get_graphs()
        nx_train_g = nx.Graph(dgl.to_networkx(dgl.to_homogeneous(dgl.node_type_subgraph(self.train_g, ntypes=['user']))))

        if self.bin_sep_lst is not None:
            train_e_id_lst, val_e_id_lst, test_e_id_lst = self.get_bins_eid_lst()
            self.bins_id_lst = val_e_id_lst

        self.cur_e = 1
        self.step('evaluate', message_g=nx_train_g, pred_g=self.val_pred_g)
        self.metric.get_batch_metrics()
        metric_str = f'Evaluate Epoch: \n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self._log(metric_str))

        if self.bin_sep_lst is not None:
            self.bins_id_lst = test_e_id_lst
        self.step('test', message_g=nx_train_g, pred_g=self.test_pred_g)
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self._log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)

