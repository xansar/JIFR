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
            # 采样
            input_nodes, pos_pred_g, neg_pred_g, blocks = inputs['graphs']
            if isinstance(input_nodes, dict):
                input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
            else:
                input_nodes = input_nodes.to(self.device)
            message_g = [b.to(self.device) for b in blocks]
            pos_pred_g = pos_pred_g.to(self.device)
            neg_pred_g = neg_pred_g.to(self.device)
            cur_step = inputs['cur_step']

            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(
                message_g,
                pos_pred_g,
                neg_pred_g,
                input_nodes=input_nodes
            )
            pos_pred, neg_pred, reg_loss, link_loss = output
            neg_pred = neg_pred.reshape(-1, self.train_neg_num)
            rate_loss = self.loss_func(pos_pred, neg_pred)
            loss = rate_loss + link_loss
            loss.backward()
            self.optimizer.step()
            return loss.item(), rate_loss.item(), reg_loss.item(), link_loss.item()

        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                # 采样
                input_nodes, output_nodes, blocks = inputs['graphs']
                if isinstance(input_nodes, dict):
                    input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
                else:
                    input_nodes = input_nodes.to(self.device)

                output_nodes = {k: v.to(self.device) for k, v in output_nodes.items()}
                message_g = [b.to(self.device) for b in blocks]

                self.model.eval()
                batch_users, rating_k, gt = \
                    self.get_full_ratings(
                        batch_users=output_nodes['user'],
                        message_g=message_g,
                        input_nodes=input_nodes,
                        mode=mode
                    )
                is_test = True if mode == 'test' else False
                self.metric.compute_metrics(batch_users.cpu(), rating_k, gt, task=self.task, is_test=is_test)
                return torch.nan, torch.nan, torch.nan, torch.nan
        else:
            raise ValueError("Wrong Mode")

    def train(self):
        loss_name = ['loss', 'rate_loss', 'reg_loss', 'link_loss']
        return self._train(loss_name)
