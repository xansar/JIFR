#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   NJBPTrainer.py
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

from .BaseTrainer import BaseTrainer

class NJBPTrainer(BaseTrainer):
    def __init__(self, config, trial=None):
        super(NJBPTrainer, self).__init__(config, trial)
        self.prepare_for_link_reg_loss()


    def prepare_for_link_reg_loss(self):
        self.omega = 2
        import json, os
        path = os.path.join('./data', self.data_name, 'splited_data', 'cache', self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(os.path.join(path, f'shortest_length_less_{self.omega}.json')):
            with open(os.path.join(path, f'shortest_length_less_{self.omega}.json'), 'r') as f:
                self.lengths = json.load(f)
        else:
            social_g = self.g.edge_type_subgraph(['trust', 'trusted-by'])
            nx_social_g = dgl.to_homogeneous(social_g).to_networkx()
            self.lengths = dict(nx.all_pairs_shortest_path_length(nx_social_g, cutoff=self.omega))
            self.lengths = {int(k): {int(_k): int(_v) for _k, _v in v.items()} for k, v in self.lengths.items()}
            with open(os.path.join(path, f'shortest_length_less_{self.omega}.json'), 'w') as f:
                json.dump(self.lengths, f, indent=2)


    def get_Y(self, u_tensor, v_tensor):
        # u_tensor: tensor, v_tensor: tensor
        ## 计算最短路径
        num = u_tensor.shape[0]
        lengths_lst = []
        for i in range(num):
            u = u_tensor[i].item()
            v = v_tensor[i].item()
            try:
                shortest_path_length = self.lengths[str(u)][str(v)]
            except KeyError:
                shortest_path_length = 0
            lengths_lst.append(shortest_path_length)
        lengths = torch.tensor(lengths_lst, dtype=torch.float, device=u_tensor.device)
        # 公式9第一个情况
        mask_1 = (lengths < self.omega + 1) & (lengths != 0)
        # 公式9第二个情况
        mask_2 = lengths > self.omega
        assert (mask_1 & mask_2).sum() == 0
        lengths[mask_1] = self.omega + 1 - lengths[mask_1]
        lengths[mask_2] = 0
        return lengths

    def compute_Y(self, g):
        if '_ID' in g.ndata.keys():
            id_map = g.ndata['_ID']['user']
            u, v = g.edges(etype=('user', 'trust', 'user'))
            u = id_map[u]
            v = id_map[v]
        else:
            u, v = g.edges(etype=('user', 'trust', 'user'))
        Y = self.get_Y(u, v)
        return Y / self.omega

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if self.task == 'Rate':
            etype = ('user', 'rate', 'item')
        elif self.task == 'Link':
            etype = ('user', 'trust', 'user')
        else:
            etype = [('user', 'rate', 'item'), ('user', 'trust', 'user')]

        if mode == 'train':
            if isinstance(inputs['graphs'], int):
                train_pos_g = self.message_g    # 全图
            else:
                train_pos_g = inputs['graphs'].to(self.device)  # 采样
            cur_step = inputs['cur_step']
            train_neg_g = self.construct_negative_graph(train_pos_g, self.train_neg_num, etype=etype)
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(
                train_pos_g,
                train_pos_g,
                train_neg_g,
            )
            if len(output) == 2:
                pos_pred, neg_pred = output
                neg_pred = neg_pred.reshape(-1, self.train_neg_num)
                loss = self.loss_func(pos_pred, neg_pred)
                loss.backward()
                self.optimizer.step()
                return loss.item()
            else:
                pos_rate_pred, neg_rate_pred, pos_link_pred, neg_link_pred = output
                rate_loss = self.loss_func(pos_rate_pred, torch.ones_like(pos_rate_pred, device=pos_rate_pred.device))\
                            + self.loss_func(neg_rate_pred, torch.zeros_like(neg_rate_pred, device=neg_rate_pred.device))
                pos_Y = torch.ones(train_pos_g.num_edges(etype=('user', 'trust', 'user')),
                                   dtype=torch.float, device=train_pos_g.device)
                # 负样本需要计算
                neg_Y = self.compute_Y(train_neg_g)
                link_loss = torch.mean(torch.square(pos_Y - pos_link_pred.reshape(-1))) + \
                            torch.mean(torch.square(neg_Y - neg_link_pred.reshape(-1)))
                loss = rate_loss + link_loss
                loss.backward()
                self.optimizer.step()
                return loss.item(), rate_loss.item(), link_loss.item()
        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                if isinstance(inputs['graphs'], int):
                    # 全图
                    message_g = self.message_g
                    pos_pred_g = self.val_pred_g if mode == 'evaluate' else self.test_pred_g
                    neg_pred_g = self.construct_negative_graph(pos_pred_g, self.neg_num, etype=etype)
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
                output = self.model(
                    message_g,
                    pos_pred_g,
                    neg_pred_g,
                    input_nodes
                )
                if len(output) == 2:
                    pos_pred, neg_pred = output
                    neg_pred = neg_pred.reshape(-1, self.neg_num)
                    loss = self.loss_func(pos_pred, neg_pred)
                    self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                    return loss.item()
                else:
                    pos_rate_pred, neg_rate_pred, pos_link_pred, neg_link_pred = output
                    if len(pos_rate_pred) != 0:
                        rate_loss = self.loss_func(pos_rate_pred,
                                                   torch.ones_like(pos_rate_pred, device=pos_rate_pred.device)) \
                                    + self.loss_func(neg_rate_pred,
                                                     torch.zeros_like(neg_rate_pred, device=neg_rate_pred.device))
                        neg_rate_pred = neg_rate_pred.reshape(-1, self.neg_num)
                        self.metric.compute_metrics(pos_rate_pred.cpu(), neg_rate_pred.cpu(), task='Rate')
                    else:
                        rate_loss = torch.tensor(-1, dtype=torch.float, device=pos_rate_pred.device)
                    if len(pos_link_pred) != 0:
                        # 正样本距离都是1
                        pos_Y = torch.ones(pos_pred_g.num_edges(etype=('user', 'trust', 'user')),
                                           dtype=torch.float, device=pos_pred_g.device)
                        # 负样本需要计算
                        neg_Y = self.compute_Y(neg_pred_g)
                        link_loss = torch.mean(torch.square(pos_Y - pos_link_pred.reshape(-1))) + \
                                    torch.mean(torch.square(neg_Y - neg_link_pred.reshape(-1)))
                        neg_link_pred = neg_link_pred.reshape(-1, self.neg_num)
                        self.metric.compute_metrics(pos_link_pred.cpu(), neg_link_pred.cpu(), task='Link')
                    else:
                        link_loss = torch.tensor(-1, dtype=torch.float, device=pos_link_pred.device)
                    if link_loss == -1:
                        loss = rate_loss
                    elif rate_loss == -1:
                        loss = link_loss
                    else:
                        loss = rate_loss + link_loss
                    return loss.item(), rate_loss.item(), link_loss.item()
        else:
            raise ValueError("Wrong Mode")

