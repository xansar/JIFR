#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   TrustSVDTrainer.py
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

class TrustSVDTrainer(BaseTrainer):
    def __init__(self, config, trial=None):
        super(TrustSVDTrainer, self).__init__(config, trial=trial)

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            if isinstance(inputs['graphs'], int):
                train_pos_g = self.message_g  # 全图
            else:
                train_pos_g = inputs['graphs'].to(self.device)  # 采样
            cur_step = inputs['cur_step']
            train_neg_g = self.construct_negative_graph(train_pos_g, self.train_neg_num, etype=('user', 'rate', 'item'))
            self.model.train()
            self.optimizer.zero_grad()
            pos_pred, neg_pred, reg_loss, link_loss = self.model(
                train_pos_g,
                train_pos_g,
                train_neg_g,
            )
            neg_pred = neg_pred.reshape(-1, self.train_neg_num)
            rate_loss = self.loss_func(pos_pred, neg_pred)
            loss = rate_loss + link_loss
            loss.backward()
            self.optimizer.step()
            return loss.item(), rate_loss.item(), reg_loss.item(), link_loss.item()
        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                if isinstance(inputs['graphs'], int):
                    # 全图
                    message_g = self.message_g
                    pos_pred_g = self.val_pred_g if mode == 'evaluate' else self.test_pred_g
                    neg_pred_g = self.construct_negative_graph(pos_pred_g, self.neg_num, etype=('user', 'rate', 'item'))
                    input_nodes = None
                else:
                    # 采样
                    input_nodes, pos_pred_g, neg_pred_g, blocks = inputs['graphs']
                    if isinstance(input_nodes, dict):
                        input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
                    else:
                        input_nodes = input_nodes.to(self.device)
                    message_g = [b.to(self.device) for b in blocks]
                    pos_pred_g = pos_pred_g.to(self.device)
                    neg_pred_g = neg_pred_g.to(self.device)

                self.model.eval()
                pos_pred, neg_pred, reg_loss, link_loss = self.model(
                    message_g,
                    pos_pred_g,
                    neg_pred_g,
                    input_nodes
                )
                neg_pred = neg_pred.reshape(-1, self.neg_num)
                rate_loss = self.loss_func(pos_pred, neg_pred)
                # loss = rate_loss + reg_loss + link_loss
                loss = rate_loss + link_loss
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                return loss.item(), rate_loss.item(), reg_loss.item(), link_loss.item()
        else:
            raise ValueError("Wrong Mode")

    def train(self):
        loss_name = ['loss', 'rate_loss', 'reg_loss', 'link_loss']
        return self._train(loss_name)
