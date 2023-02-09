#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   CrossFireModel.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/4 19:19   zxx      1.0         None
"""

# import lib
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl.function as fn

from .utils import init_weights, BaseModel


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class CrossFireModel(BaseModel):
    def __init__(self, config, etype=None):
        super(CrossFireModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.embedding = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        # self.Q = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        # self.P = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))
        self.pred = HeteroDotProductPredictor()
        init_weights(self.modules())

    def get_final_embeddings(self, output_nodes, etype):
        if etype == 'rate':
            # u_embed = torch.matmul(self.res_embedding['user'][output_nodes], self.Q)
            u_embed = self.res_embedding['user'][output_nodes]
            v_embed = self.res_embedding['item']
        else:
            # u_embed = torch.matmul(self.res_embedding['user'][output_nodes], self.P)
            u_embed = self.res_embedding['user'][output_nodes]
            v_embed = self.res_embedding['user']
        return u_embed, v_embed, None, None

    def compute_final_embeddings(self, message_g=None, idx=None):
        self.res_embedding = self.embedding.weight

    def forward(self, message_g, pos_pred_g, neg_pred_g, input_nodes=None):
        res_embedding = self.embedding(input_nodes)

        dst_user = message_g[0].dstnodes(ntype='user')
        dst_item = message_g[0].dstnodes(ntype='item')
        res_embedding = {
            # 'user': torch.matmul(res_embedding['user'][dst_user], self.Q),
            'user': res_embedding['user'][dst_user],
            'item': res_embedding['item'][dst_item]
        }
        pos_rate_score = self.pred(pos_pred_g, res_embedding, 'rate')
        neg_rate_score = self.pred(neg_pred_g, res_embedding, 'rate')
        # res_embedding = {
        #     # 'user': torch.matmul(res_embedding['user'][dst_user], self.P),
        #     'user': res_embedding['user'][dst_user],
        #     'item': res_embedding['item'][dst_item]
        # }
        pos_link_score = self.pred(pos_pred_g, res_embedding, 'trust')
        neg_link_score = self.pred(neg_pred_g, res_embedding, 'trust')
        return pos_rate_score, neg_rate_score, pos_link_score, neg_link_score