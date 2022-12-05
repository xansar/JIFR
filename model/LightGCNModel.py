#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LightGCNModel.py    
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


class LightGCNModel(nn.Module):
    def __init__(self, config, rel_names):
        super(LightGCNModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.layer_num = eval(config['MODEL']['gcn_layer_num'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.embedding = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(
                 dglnn.HeteroGraphConv({
                    rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='both', weight=False, bias=False)
                    for rel in rel_names
                })
            )
        self.pred = HeteroDotProductPredictor()

    def forward(self, messege_g, pos_pred_g, neg_pred_g, input_nodes=None):
        # idx = {ntype: messege_g.nodes(ntype) for ntype in messege_g.ntypes}
        # res_embedding = self.embedding(idx)
        # for i, layer in enumerate(self.layers):
        #     if i == 0:
        #         embeddings = layer(messege_g, res_embedding)
        #     else:
        #         embeddings = layer(messege_g, embeddings)
        #     # print(embeddings)
        #     # print(res_embedding['user'].shape, embeddings['user'].shape)
        #     # print(res_embedding['item'].shape, embeddings['item'].shape)
        #     res_embedding['user'] = res_embedding['user'] + embeddings['user'] * (1 / (i + 2))
        #     res_embedding['item'] = res_embedding['item'] + embeddings['item'] * (1 / (i + 2))
        # pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        # neg_score = self.pred(neg_pred_g, res_embedding, 'rate')

        if input_nodes is None:
            idx = {ntype: messege_g.nodes[ntype].data['_ID'] for ntype in messege_g.ntypes}
            res_embedding = self.embedding(idx)
            for i, layer in enumerate(self.layers):
                if i == 0:
                    embeddings = layer(messege_g, res_embedding)
                else:
                    embeddings = layer(messege_g, embeddings)
                res_embedding['user'] = res_embedding['user'] + embeddings['user']
                res_embedding['item'] = res_embedding['item'] + embeddings['item']
            res_embedding['user'] /= (1 / len(self.layers))
            res_embedding['item'] /= (1 / len(self.layers))
        else:
            original_embedding = self.embedding(input_nodes)
            dst_user = pos_pred_g.dstnodes(ntype='user')
            dst_item = pos_pred_g.dstnodes(ntype='item')
            res_embedding = {
                'user': original_embedding['user'][dst_user],
                'item': original_embedding['item'][dst_item]
            }
            for i in range(1, len(messege_g) + 1):
                blocks = messege_g[-i:]
                for j in range(i):
                    layer = self.layers[j]
                    if j == 0:
                        embeddings = layer(blocks[j], original_embedding)
                    else:
                        embeddings = layer(blocks[j], embeddings)
                res_embedding['user'] = res_embedding['user'] + embeddings['user']
                res_embedding['item'] = res_embedding['item'] + embeddings['item']
            res_embedding['user'] /= (1 / len(self.layers))
            res_embedding['item'] /= (1 / len(self.layers))
        pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        neg_score = self.pred(neg_pred_g, res_embedding, 'rate')
        return pos_score, neg_score