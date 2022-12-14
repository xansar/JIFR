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


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class MFModel(nn.Module):
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
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, messege_g, pos_pred_g, neg_pred_g, input_nodes=None):
        if input_nodes is None:
            # 训练时，在图上训练，不需要区分block
            idx = {ntype: messege_g.nodes[ntype].data['_ID'] for ntype in messege_g.ntypes}
            if self.task == 'Link':
                etype = 'trust'
                res_embedding = self.embedding(idx)['user']
            elif self.task == 'Rate':
                etype = 'rate'
                res_embedding = self.embedding(idx)
        else:
            # 测试时需要注意block
            if self.task == 'Link':
                res_embedding = self.embedding({'user': input_nodes})
                etype = 'trust'
                dst_user = messege_g[0].dstnodes(ntype='user')
                res_embedding = res_embedding['user'][dst_user]
            elif self.task == 'Rate':
                res_embedding = self.embedding(input_nodes)
                etype = 'rate'
                dst_user = messege_g[0].dstnodes(ntype='user')
                dst_item = messege_g[0].dstnodes(ntype='item')
                res_embedding = {
                    'user': res_embedding['user'][dst_user],
                    'item': res_embedding['item'][dst_item]
                }

        pos_score = self.pred(pos_pred_g, res_embedding, etype)
        neg_score = self.pred(neg_pred_g, res_embedding, etype)
        return pos_score, neg_score