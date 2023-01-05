#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   DiffnetPPModel.py
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
    
    
class DiffusionLayer(nn.Module):
    def __init__(self, rel_names, embedding_size, num_heads):
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        super(DiffusionLayer, self).__init__()
        self.interest_diffusion_layer = dglnn.HeteroGraphConv({
            rel: dglnn.GATv2Conv(self.embedding_size, self.embedding_size, num_heads=self.num_heads)
            for rel in rel_names
        })

        self.influence_diffusion_layer = dglnn.HeteroGraphConv({
            rel: dglnn.GATv2Conv(self.embedding_size, self.embedding_size, num_heads=self.num_heads)
            for rel in rel_names
        })
        self.att_score_influence = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            # nn.BatchNorm1d(self.embedding_size),
            nn.Linear(self.embedding_size, 1),
            # nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )
        self.att_score_interest = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            # nn.BatchNorm1d(self.embedding_size),
            nn.Linear(self.embedding_size, 1),
            # nn.BatchNorm1d(1),
            nn.LeakyReLU()
        )

    def forward(self, g, embedding):
        src_user = g.srcnodes(ntype='user')
        src_item = g.srcnodes(ntype='item')
        dst_user = g.dstnodes(ntype='user')
        dst_item = g.dstnodes(ntype='item')
        # item
        src = {'user': embedding['user'][src_user]}
        dst = {'item': embedding['item'][dst_item]}
        item_embedding = self.interest_diffusion_layer(g, (src, dst))['item'].squeeze(1) + dst['item']    # 'item': embed

        # user
        ## item->user
        src = {'item': embedding['item'][src_item]}
        dst = {'user': embedding['user'][dst_user]}
        q_hair = self.interest_diffusion_layer(g, (src, dst))['user'].squeeze(1)   # interest
        ## user->user
        src = {'user': embedding['user'][src_user]}
        dst = {'user': embedding['user'][dst_user]}
        p_hair = self.influence_diffusion_layer(g, (src, dst))['user'].squeeze(1)  # influence
        ## att
        influence_att_score = self.att_score_influence(torch.cat([embedding['user'][dst_user], p_hair], dim=1))
        interest_att_score = self.att_score_interest(torch.cat([embedding['user'][dst_user], q_hair], dim=1))
        gamma = torch.softmax(torch.cat([influence_att_score, interest_att_score], dim=1), dim=1)
        user_embedding = gamma[:, 0].unsqueeze(1) * p_hair + gamma[:, 1].unsqueeze(1) * q_hair + embedding['user'][dst_user]
        return {
            'user': user_embedding,
            'item': item_embedding
        }

class ActivatedHeteroLinear(nn.Module):
    def __init__(self, input_size: dict, hidden_size: int, out_put_size: int, layer_num=2):
        super(ActivatedHeteroLinear, self).__init__()
        self.hetero_liner = nn.ModuleList()
        self.hidden_size_dict = {k: hidden_size for k in input_size.keys()}
        for i in range(layer_num):
            if i == 0:
                self.hetero_liner.append(
                    dglnn.HeteroLinear(input_size, hidden_size)
                )
            elif i < layer_num - 1:
                self.hetero_liner.append(
                    dglnn.HeteroLinear(self.hidden_size_dict, hidden_size)
                )
            else:
                self.hetero_liner.append(
                    dglnn.HeteroLinear(self.hidden_size_dict, out_put_size)
                )
        self.hetero_batch_norm = nn.ModuleDict({
            k: nn.BatchNorm1d(out_put_size)
            for k in input_size.keys()
        })
        self.hetero_activation = ({
            k: nn.LeakyReLU()
            for k in input_size.keys()
        })

    def forward(self, x: dict):
        assert x.keys() == self.hidden_size_dict.keys()
        # linear
        for i, layer in enumerate(self.hetero_liner):
            x = layer(x)
        # batchnorm + activation
        for k in x.keys():
            x[k] = self.hetero_activation[k](self.hetero_batch_norm[k](x[k]))
        return x


class DiffnetPPModel(BaseModel):
    def __init__(self, config, rel_names, is_feature=False):
        super(DiffnetPPModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.mlp_layer_num = eval(config['MODEL']['mlp_layer_num'])
        self.gcn_layer_num = eval(config['MODEL']['gcn_layer_num'])
        self.num_heads = eval(config['MODEL']['num_heads'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.embedding = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        input_f_size = 2 * self.embedding_size if is_feature else self.embedding_size

        self.fusion_layer = ActivatedHeteroLinear(
            {'user': input_f_size, 'item': input_f_size},
            input_f_size,
            self.embedding_size,
            layer_num=self.mlp_layer_num
        )

        self.diffusion_layers = nn.ModuleList()
        for i in range(self.gcn_layer_num):
            self.diffusion_layers.append(DiffusionLayer(rel_names, self.embedding_size, self.num_heads))

        self.pred = HeteroDotProductPredictor()
        init_weights(self.modules())

    def compute_final_embeddings(self, message_g, idx=None):
        mode = 'train'
        if idx is None:
            mode = 'evaluate'
            idx = {ntype: message_g.nodes(ntype=ntype) for ntype in message_g.ntypes}
        res_embedding = self.embedding(idx)
        for i, layer in enumerate(self.diffusion_layers):
            if i == 0:
                embeddings = layer(message_g, res_embedding)
            else:
                embeddings = layer(message_g, embeddings)
            res_embedding['user'] = torch.cat([res_embedding['user'], embeddings['user']], dim=1)
            res_embedding['item'] = torch.cat([res_embedding['item'], embeddings['item']], dim=1)
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
            # res_embedding = self.fusion_layer(self.embedding(idx))
            res_embedding = self.embedding(idx)
            for i, layer in enumerate(self.diffusion_layers):
                if i == 0:
                    embeddings = layer(message_g, res_embedding)
                else:
                    embeddings = layer(message_g, embeddings)
                res_embedding['user'] = torch.cat([res_embedding['user'], embeddings['user']], dim=1)
                res_embedding['item'] = torch.cat([res_embedding['item'], embeddings['item']], dim=1)
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
                    layer = self.diffusion_layers[j]
                    if j == 0:
                        embeddings = layer(blocks[j], original_embedding)
                    else:
                        embeddings = layer(blocks[j], embeddings)
                res_embedding['user'] = torch.cat([res_embedding['user'], embeddings['user']], dim=1)
                res_embedding['item'] = torch.cat([res_embedding['item'], embeddings['item']], dim=1)
        pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        neg_score = self.pred(neg_pred_g, res_embedding, 'rate')
        return pos_score, neg_score