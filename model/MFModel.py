#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MFModel.py
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


class MFModel(BaseModel):
    def __init__(self, config, etype=None):
        super(MFModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.embedding = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        self.pred = HeteroDotProductPredictor()
        init_weights(self.modules())

    def compute_final_embeddings(self, message_g):
        self.res_embedding = self.embedding.weight

    def forward(self, message_g, pos_pred_g, neg_pred_g, input_nodes=None):
        if input_nodes is None:
            # 训练时，在图上训练，不需要区分block
            if '_ID' in message_g.ndata.keys():
                # 子图采样的情况
                idx = {ntype: message_g.nodes[ntype].data['_ID'] for ntype in message_g.ntypes}
            else:
                # 全图的情况
                idx = {ntype: message_g.nodes(ntype=ntype) for ntype in message_g.ntypes}
            if self.task == 'Link':
                etype = 'trust'
                res_embedding = self.embedding(idx)['user']
            elif self.task == 'Rate':
                etype = 'rate'
                res_embedding = self.embedding(idx)
        else:
            # 测试时需要注意block
            if self.task == 'Link':
                res_embedding = self.embedding({'user': input_nodes})   # block的情况
                etype = 'trust'
                dst_user = message_g[0].dstnodes(ntype='user')
                res_embedding = res_embedding['user'][dst_user]
            elif self.task == 'Rate':
                res_embedding = self.embedding(input_nodes)
                etype = 'rate'
                dst_user = message_g[0].dstnodes(ntype='user')
                dst_item = message_g[0].dstnodes(ntype='item')
                res_embedding = {
                    'user': res_embedding['user'][dst_user],
                    'item': res_embedding['item'][dst_item]
                }

        pos_score = self.pred(pos_pred_g, res_embedding, etype)
        neg_score = self.pred(neg_pred_g, res_embedding, etype)
        return pos_score, neg_score