#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   FLGNModel.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/4 19:19   zxx      1.0         None
"""

# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FLGNModel(BaseModel):
    def __init__(self, config, rel_names):
        super(FLGNModel, self).__init__()
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
        init_weights(self.modules())

    def compute_final_embeddings(self, message_g, idx=None):
        mode = 'train'
        if idx is None:
            mode = 'evaluate'
            idx = {ntype: message_g.nodes(ntype=ntype) for ntype in message_g.ntypes}
        res_embedding = self.embedding(idx)

        for i, layer in enumerate(self.layers):
            if i == 0:
                cur_embed = res_embedding
            else:
                cur_embed = embeddings
            # item->user
            src = {'item': cur_embed['item']}
            dst = {'user': cur_embed['user']}
            embeddings = layer(message_g, (src, dst))
            # user->item
            src = {'user': cur_embed['user']}
            dst = {'item': cur_embed['item']}
            embeddings.update(layer(message_g, (src, dst)))

            # user->user
            src = {'user': cur_embed['user']}
            dst = {'user': cur_embed['user']}
            social_embedding = layer(message_g, (src, dst))['user']
            pref_embedding = embeddings['user']
            embeddings['user'] = social_embedding + pref_embedding

            res_embedding['user'] = res_embedding['user'] + embeddings['user']
            res_embedding['item'] = res_embedding['item'] + embeddings['item']

        res_embedding['user'] /= len(self.layers) + 1
        res_embedding['item'] /= len(self.layers) + 1
        if mode == 'evaluate':
            self.res_embedding = res_embedding
        else:
            return res_embedding


    def forward(self, message_g, pos_pred_g, neg_pred_g, input_nodes=None):
        if input_nodes is None:
            if '_ID' in message_g.ndata.keys():
                # 子图采样的情况
                idx = {ntype: message_g.nodes[ntype].data['_ID'] for ntype in message_g.ntypes}
            else:
                # 全图的情况
                idx = {ntype: message_g.nodes(ntype=ntype) for ntype in message_g.ntypes}
            res_embedding = self.compute_final_embeddings(message_g, idx)
        else:
            original_embedding = self.embedding(input_nodes)
            dst_user = pos_pred_g.dstnodes(ntype='user')
            dst_item = pos_pred_g.dstnodes(ntype='item')
            res_embedding = {
                'user': original_embedding['user'][dst_user],
                'item': original_embedding['item'][dst_item]
            }
            for i in range(1, len(message_g) + 1):
                blocks = message_g[-i:]
                for j in range(i):
                    layer = self.layers[j]
                    embed = {
                        'user': original_embedding['user'][blocks[j].srcnodes('user')],
                        'item': original_embedding['item'][blocks[j].srcnodes('item')],
                    }
                    if j == 0:
                        cur_embed = embed
                    else:
                        cur_embed = embeddings
                    # item->user
                    src = {'item': cur_embed['item']}
                    dst = {'user': cur_embed['user']}
                    embeddings = layer(blocks[j], (src, dst))
                    # user->item
                    src = {'user': cur_embed['user']}
                    dst = {'item': cur_embed['item']}
                    embeddings.update(layer(blocks[j], (src, dst)))
                    # user->user
                    src = {'user': cur_embed['user']}
                    dst = {'user': cur_embed['user']}
                    social_embedding = layer(blocks[j], (src, dst))['user']
                    # 去掉mlp和正则化，效果变好了
                    pref_embedding = embeddings['user']
                    embeddings['user'] = social_embedding + pref_embedding

                res_embedding['user'] = res_embedding['user'] + embeddings['user']
                res_embedding['item'] = res_embedding['item'] + embeddings['item']
            res_embedding['user'] /= len(self.layers) + 1
            res_embedding['item'] /= len(self.layers) + 1
        pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        neg_score = self.pred(neg_pred_g, res_embedding, 'rate')
        return pos_score, neg_score