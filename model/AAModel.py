#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   AAModel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/24 14:09   zxx      1.0         None
"""

# import lib
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import networkx as nx
from math import log

class AAModel(nn.Module):
    def __init__(self, config):
        super(AAModel, self).__init__()
        self.params = nn.Embedding(1, 1)    # 没用，单纯为了适配框架

    def forward(self, messeage_g, pos_pred_g_edges_lst, neg_pred_g_edges_lst):
        pos_score = []
        for u, v in pos_pred_g_edges_lst:
            sum_log = 0.
            for w in nx.common_neighbors(messeage_g, u, v):
                sum_log += 1 / log(messeage_g.degree(w))
            pos_score.append([u, v, sum_log])
            # pos_score.append([u, v, sum([1 / log(messeage_g.degree(w)) for w in nx.common_neighbors(messeage_g, u, v)])])
        neg_score = []
        for u, v in neg_pred_g_edges_lst:
            sum_log = 0.
            for w in nx.common_neighbors(messeage_g, u, v):
                sum_log += 1 / log(messeage_g.degree(w))
            neg_score.append([u, v, sum_log])
            # neg_score.append([u, v, sum([1 / log(messeage_g.degree(w)) for w in nx.common_neighbors(messeage_g, u, v)])])
        pos_score = torch.tensor(pos_score)[:, 2].reshape(-1, 1)
        neg_score = torch.tensor(neg_score)[:, 2].reshape(-1, 1)
        return pos_score, neg_score
