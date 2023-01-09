#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   SNCLTrainer.py
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

class SNCLTrainer(BaseTrainer):
    def __init__(self, config, trial=None):
        super(SNCLTrainer, self).__init__(config, trial=trial)

    def step(self, mode='train', **inputs):
        # 模型单步计算
        etype = ('user', 'rate', 'item')

        if mode == 'train':
            if isinstance(inputs['graphs'], int):
                message_g = self.message_g    # 全图
                pos_pred_g = message_g
                neg_pred_g = self.construct_negative_graph(message_g, self.train_neg_num, etype=etype)
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

            # if isinstance(inputs['graphs'], int):
            #     train_pos_g = self.message_g    # 全图
            # else:
            #     train_pos_g = inputs['graphs'].to(self.device)  # 采样
            #
            # cur_step = inputs['cur_step']
            # train_neg_g = self.construct_negative_graph(train_pos_g, self.train_neg_num, etype=etype)
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(
                message_g,
                pos_pred_g,
                neg_pred_g,
                input_nodes=input_nodes
            )
            pos_pred, neg_pred, ssl_loss, proto_nce_loss = output
            neg_pred = neg_pred.reshape(-1, self.train_neg_num)
            loss = self.loss_func(pos_pred, neg_pred)
            total_loss = loss + ssl_loss + proto_nce_loss
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item(), loss.item(), ssl_loss.item(), proto_nce_loss.item()

        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
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
                self.metric.compute_metrics(batch_users.cpu(), rating_k, gt, task=self.task)
                return torch.nan, torch.nan, torch.nan, torch.nan
                #
                # if isinstance(inputs['graphs'], int):
                #     # 全图
                #     message_g = self.message_g
                #     pos_pred_g = self.val_pred_g if mode == 'evaluate' else self.test_pred_g
                #     neg_pred_g = self.construct_negative_graph(pos_pred_g, self.neg_num, etype=etype)
                #     input_nodes = None
                # else:
                #     # 采样
                #     input_nodes, pos_pred_g, neg_pred_g, blocks = inputs['graphs']
                #     if isinstance(input_nodes, dict):
                #         input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
                #     else:
                #         input_nodes = input_nodes.to(self.device)
                #     message_g = [b.to(self.device) for b in blocks]
                #     pos_pred_g = pos_pred_g.to(self.device)
                #     neg_pred_g = neg_pred_g.to(self.device)
                #
                # self.model.eval()
                # output = self.model(
                #     message_g,
                #     pos_pred_g,
                #     neg_pred_g,
                #     input_nodes
                # )
                #
                # pos_pred, neg_pred, ssl_loss, proto_nce_loss = output
                # neg_pred = neg_pred.reshape(-1, self.neg_num)
                # loss = self.loss_func(pos_pred, neg_pred)
                # total_loss = loss + ssl_loss + proto_nce_loss
                # self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                # return total_loss.item(), loss.item(), ssl_loss.item(), proto_nce_loss
        else:
            raise ValueError("Wrong Mode")

    def wrap_step_loop(self, mode, data_loader: dgl.dataloading.DataLoader or range, side_info, loss_name):
        all_loss_lst = [0.0 for _ in range(len(loss_name))]

        val_true_batch_cnt = [0 for _ in range(len(loss_name))]
        if self.is_log:
            bar_loader = tqdm(enumerate(data_loader), total=len(data_loader))
        else:
            bar_loader = enumerate(data_loader)
        if mode != 'train':
            self.metric.iter_step = 0
            # 计算所有user item的最终表示
            self.model.compute_final_embeddings(self.message_g)

        if mode == 'train':
            # 聚类，每个epoch做一次
            self.model.e_step()

        for i, graphs in bar_loader:
            loss_lst = self.step(
                mode=mode, graphs=graphs,
                side_info=side_info, cur_step=i)
            for j in range(len(loss_name)):
                if len(loss_name) == 1:
                    loss_lst = [loss_lst]
                # 因为如果需要测试两种边，dataloader会依次读入两种类型，如果此时将没有出现的loss记为0，会导致数据不准
                # 做法是，rate批次设置link loss为0，link批次设置rate loss为0
                # 因此，需要记录rate批次和link批次的次数，一个epoch完成后再进行校正
                if loss_lst[j] != -1:
                    val_true_batch_cnt[j] += 1
                    all_loss_lst[j] += loss_lst[j]
                else:
                    all_loss_lst[j] += 0
            if mode != 'train':
                self.metric.iter_step += 1

        for j in range(len(loss_name)):
            # if val_true_batch_cnt[j] != 0:
            #     # 说明loss需要校正
            #     all_loss_lst[j] /= len(data_loader)
            # all_loss_lst[j] /= len(data_loader)

            # 如果出现rate和link的数量和不等于total的，不用担心，因为会有一个batch包含两类边
            all_loss_lst[j] /= val_true_batch_cnt[j]
        return all_loss_lst

    def train(self):
        loss_name = ['Total Loss', 'Loss', 'Structure Loss', 'Proto NCE Loss']
        return self._train(loss_name)