#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   DiffnetPPTrainer.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/4 21:19   zxx      1.0         None
"""

# import lib

import dgl
import torch
import numpy as np
from tqdm import tqdm, trange
import os

from .BaseTrainer import BaseTrainer

class DiffnetPPTrainer(BaseTrainer):
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
        super(DiffnetPPTrainer, self).__init__(config)
        # 读取数据
        self.dataset = dataset
        self.g = dataset[0]
        self.train_size = dataset.train_size
        self.val_size = dataset.val_size
        self.train_link_size = dataset.train_link_size
        self.val_link_size = dataset.val_link_size

        # 读取训练有关配置
        self.user_num = eval(config['MODEL']['pred_user_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
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
            ('user', 'rate', 'item'): range(self.train_size),
            ('item', 'rated-by', 'user'): range(self.train_size),
            ('user', 'trust', 'user'): range(self.train_link_size),
            ('user', 'trusted-by', 'user'): range(self.train_link_size)
        }
        train_g = dgl.edge_subgraph(self.g, train_edges, relabel_nodes=False)

        val_edges = {
            ('user', 'rate', 'item'): range(self.train_size, self.train_size + self.val_size),
            ('item', 'rated-by', 'user'): range(self.train_size, self.train_size + self.val_size),
            ('user', 'trust', 'user'): range(self.train_link_size, self.train_link_size + self.val_link_size),
            ('user', 'trusted-by', 'user'): range(self.train_link_size, self.train_link_size + self.val_link_size)
        }
        val_pred_g = dgl.edge_subgraph(self.g, val_edges, relabel_nodes=False)

        test_edges = {
            ('user', 'rate', 'item'): range(self.train_size + self.val_size, self.g.num_edges(('user', 'rate', 'item'))),
            ('item', 'rated-by', 'user'): range(self.train_size + self.val_size, self.g.num_edges(('item', 'rated-by', 'user'))),
            ('user', 'trust', 'user'): range(self.train_link_size + self.val_link_size, self.g.num_edges(('user', 'trusted-by', 'user'))),
            ('user', 'trusted-by', 'user'): range(self.train_link_size + self.val_link_size, self.g.num_edges(('user', 'trusted-by', 'user')))
        }
        test_pred_g = dgl.edge_subgraph(self.g, test_edges, relabel_nodes=False)
        # val_g = dgl.edge_subgraph(self.g, range(self.train_size + self.val_size), relabel_nodes=False)

        return train_g, val_pred_g, test_pred_g

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            train_pos_g = inputs['train_pos_g']
            train_neg_g = self.construct_negative_graph(train_pos_g, self.train_neg_num, etype=('user', 'rate', 'item'))
            self.model.train()
            self.optimizer.zero_grad()
            pos_pred, neg_pred = self.model(
                train_pos_g,
                train_pos_g,
                train_neg_g
            )
            neg_pred = neg_pred.reshape(-1, self.train_neg_num)
            loss = self.loss_func(pos_pred, neg_pred)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                message_g = inputs['message_g']
                pred_g = inputs['pred_g']
                neg_g = self.construct_negative_graph(pred_g, self.neg_num, etype=('user', 'rate', 'item'))
                self.model.eval()
                pos_pred, neg_pred = self.model(
                    message_g,
                    pred_g,
                    neg_g
                )
                neg_pred = neg_pred.reshape(-1, self.neg_num)
                loss = self.loss_func(pos_pred, neg_pred)
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def train(self):
        loss_name = ['Loss']
        self._train(loss_name)
