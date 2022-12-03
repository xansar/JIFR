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
    def __init__(self, config,):
        super(DiffnetPPTrainer, self).__init__(config)

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            train_pos_g = inputs['graphs'].to(self.device)
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
                input_nodes, pos_pred_g, neg_pred_g, blocks = inputs['graphs']
                input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
                blocks = [b.to(self.device) for b in blocks]
                pos_pred_g = pos_pred_g.to(self.device)
                neg_pred_g = neg_pred_g.to(self.device)

                self.model.eval()
                pos_pred, neg_pred = self.model(
                    blocks,
                    pos_pred_g,
                    neg_pred_g,
                    input_nodes
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
