#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LightGCNTrainer.py
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

class LightGCNTrainer(BaseTrainer):
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
        super(LightGCNTrainer, self).__init__(config)
        # 读取数据
        self.dataset = dataset
        self.g = dataset[0]
        self.train_size = dataset.train_size
        self.val_size = dataset.val_size

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
            meta_etype: range(self.train_size) for meta_etype in self.g.canonical_etypes
        }
        train_g = dgl.edge_subgraph(self.g, train_edges, relabel_nodes=False)
        val_edges = {
            meta_etype: range(self.train_size, self.train_size + self.val_size) for meta_etype in self.g.canonical_etypes
        }
        val_pred_g = dgl.edge_subgraph(self.g, val_edges, relabel_nodes=False)
        test_edges = {
            meta_etype: range(self.train_size + self.val_size, self.g.num_edges(meta_etype)) for meta_etype in self.g.canonical_etypes
        }
        test_pred_g = dgl.edge_subgraph(self.g, test_edges, relabel_nodes=False)
        # val_g = dgl.edge_subgraph(self.g, range(self.train_size + self.val_size), relabel_nodes=False)

        return train_g, val_pred_g, test_pred_g

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            train_pos_g = inputs['train_pos_g']
            cur_step = inputs['cur_step']
            train_neg_g = self.construct_negative_graph(train_pos_g, self.train_neg_num, etype=('user', 'rate', 'item'))
            self.model.train()
            self.optimizer.zero_grad()
            pos_pred, neg_pred = self.model(
                train_pos_g,
                train_pos_g,
                train_neg_g
            )
            neg_pred = neg_pred.reshape(-1, self.train_neg_num)
            if self.bin_sep_lst is not None and self.is_visulized == True:
                if cur_step == 0:
                    self.log_pred_histgram(pos_pred, neg_pred, mode)

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
                if self.bin_sep_lst is not None and self.is_visulized == True:
                    self.log_pred_histgram(pos_pred, neg_pred, mode)

                loss = self.loss_func(pos_pred, neg_pred)
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def train(self):
        loss_name = ['Loss']
        self._train(loss_name)
