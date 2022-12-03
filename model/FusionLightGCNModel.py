#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   FusionLightGCNModel.py
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


class FusionLightGCNModel(nn.Module):
    def __init__(self, config, rel_names):
        super(FusionLightGCNModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.layer_num = eval(config['MODEL']['gcn_layer_num'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.embedding = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        self.preference_layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.preference_layers.append(
                 dglnn.HeteroGraphConv({
                    rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='both', weight=False, bias=False)
                    for rel in rel_names
                })
            )

        self.social_layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.social_layers.append(
                 dglnn.HeteroGraphConv({
                    rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='both', weight=False, bias=False)
                    for rel in rel_names
                })
            )
        self.user_mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_size, 2 * self.embedding_size),
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.LeakyReLU()
        )
        self.pred = HeteroDotProductPredictor()

    def forward(self, messege_g, pos_pred_g, neg_pred_g, social_network):
        idx = {ntype: messege_g.nodes(ntype) for ntype in messege_g.ntypes}
        pref_embedding = self.embedding(idx)
        social_embedding = {
            'user': torch.clone(pref_embedding['user'])
        }
        for i, layer in enumerate(self.preference_layers):
            if i == 0:
                embeddings = layer(messege_g, pref_embedding)
            else:
                embeddings = layer(messege_g, embeddings)
            # print(embeddings)
            # print(pref_embedding['user'].shape, embeddings['user'].shape)
            # print(pref_embedding['item'].shape, embeddings['item'].shape)
            pref_embedding['user'] = pref_embedding['user'] + embeddings['user'] * (1 / (i + 2))
            pref_embedding['item'] = pref_embedding['item'] + embeddings['item'] * (1 / (i + 2))

        for i, layer in enumerate(self.social_layers):
            if i == 0:
                embeddings = layer(social_network, social_embedding)
            else:
                embeddings = layer(social_network, embeddings)
            # print(embeddings)
            # print(pref_embedding['user'].shape, embeddings['user'].shape)
            # print(pref_embedding['item'].shape, embeddings['item'].shape)
            social_embedding['user'] = social_embedding['user'] + embeddings['user'] * (1 / (i + 2))

        res_embedding = {
            'user': pref_embedding['user'] + social_embedding['user'],
            'item': pref_embedding['item']
        }
        pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        neg_score = self.pred(neg_pred_g, res_embedding, 'rate')
        return pos_score, neg_score