#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   SocialLGNModel.py
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

from .utils import init_weights

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class SocialLGNModel(nn.Module):
    def __init__(self, config, rel_names):
        super(SocialLGNModel, self).__init__()
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
        self.fusion_layer_social_part = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        self.fusion_layer_pref_part = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )
        for i in range(self.layer_num):
            self.layers.append(
                 dglnn.HeteroGraphConv({
                    rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='both', weight=False, bias=False)
                    for rel in rel_names
                })
            )

        self.pred = HeteroDotProductPredictor()
        init_weights(self.modules())

    def forward(self, messege_g, pos_pred_g, neg_pred_g, input_nodes=None):
        if input_nodes is None:
            if '_ID' in messege_g.ndata.keys():
                # 子图采样的情况
                idx = {ntype: messege_g.nodes[ntype].data['_ID'] for ntype in messege_g.ntypes}
            else:
                # 全图的情况
                idx = {ntype: messege_g.nodes(ntype=ntype) for ntype in messege_g.ntypes}
            res_embedding = self.embedding(idx)

            for i, layer in enumerate(self.layers):
                if i == 0:
                    cur_embed = res_embedding
                else:
                    cur_embed = embeddings
                # item->user
                src = {'item': cur_embed['item']}
                dst = {'user': cur_embed['user']}
                embeddings = layer(messege_g, (src, dst))
                # user->item
                src = {'user': cur_embed['user']}
                dst = {'item': cur_embed['item']}
                embeddings.update(layer(messege_g, (src, dst)))

                # user->user
                src = {'user': cur_embed['user']}
                dst = {'user': cur_embed['user']}
                social_embedding = layer(messege_g, (src, dst))['user']
                ## fusion layer
                embeddings['user'] = self.fusion_layer_social_part(social_embedding)\
                                     + self.fusion_layer_pref_part(embeddings['user'])
                ## regularization
                embeddings['user'] = F.normalize(embeddings['user'])

                res_embedding['user'] = res_embedding['user'] + embeddings['user']
                res_embedding['item'] = res_embedding['item'] + embeddings['item']

            res_embedding['user'] /= len(self.layers) + 1
            res_embedding['item'] /= len(self.layers) + 1
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
                        cur_embed = original_embedding
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
                    ## fusion layer
                    embeddings['user'] = self.fusion_layer_social_part(social_embedding) \
                                         + self.fusion_layer_pref_part(embeddings['user'])
                    ## regularization
                    embeddings['user'] = F.normalize(embeddings['user'])

                res_embedding['user'] = res_embedding['user'] + embeddings['user']
                res_embedding['item'] = res_embedding['item'] + embeddings['item']
            res_embedding['user'] /= len(self.layers) + 1
            res_embedding['item'] /= len(self.layers) + 1
        pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        neg_score = self.pred(neg_pred_g, res_embedding, 'rate')
        return pos_score, neg_score
        # pref_embedding = self.embedding(idx)
        # social_embedding = {
        #     'user': torch.clone(pref_embedding['user'])
        # }
        # for i, layer in enumerate(self.preference_layers):
        #     if i == 0:
        #         embeddings = layer(messege_g, pref_embedding)
        #     else:
        #         embeddings = layer(messege_g, embeddings)
        #     # print(embeddings)
        #     # print(pref_embedding['user'].shape, embeddings['user'].shape)
        #     # print(pref_embedding['item'].shape, embeddings['item'].shape)
        #     pref_embedding['user'] = pref_embedding['user'] + embeddings['user'] * (1 / (i + 2))
        #     pref_embedding['item'] = pref_embedding['item'] + embeddings['item'] * (1 / (i + 2))
        #
        # for i, layer in enumerate(self.social_layers):
        #     if i == 0:
        #         embeddings = layer(social_network, social_embedding)
        #     else:
        #         embeddings = layer(social_network, embeddings)
        #     # print(embeddings)
        #     # print(pref_embedding['user'].shape, embeddings['user'].shape)
        #     # print(pref_embedding['item'].shape, embeddings['item'].shape)
        #     social_embedding['user'] = social_embedding['user'] + embeddings['user'] * (1 / (i + 2))
        #
        # res_embedding = {
        #     'user': pref_embedding['user'] + social_embedding['user'],
        #     'item': pref_embedding['item']
        # }
        # pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        # neg_score = self.pred(neg_pred_g, res_embedding, 'rate')
        # return pos_score, neg_score