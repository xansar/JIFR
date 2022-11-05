#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MutualRecModel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/28 18:41   zxx      1.0         None
"""

# import lib
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch
import torch.nn as nn


class SpatialAttentionLayer(nn.Module):
    def __init__(self, rel_names, embedding_size=1, num_heads=1):
        super(SpatialAttentionLayer, self).__init__()
        self.gat_layer_1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATv2Conv(embedding_size, embedding_size, num_heads=num_heads)
                for rel in rel_names
            },
            aggregate='mean'
        )
        self.gat_layer_2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATv2Conv(embedding_size, embedding_size, num_heads=num_heads)
                for rel in rel_names
            },
            aggregate='mean'
        )

        self.output = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )

    def forward(self, g, user_embed, item_embed):
        # print(h)
        u = {'user': user_embed}
        i = {'item': item_embed}
        # user->item
        rsc = u
        dst = i
        h1 = self.gat_layer_1(g, (rsc, dst))
        h1['item'] = h1['item'].squeeze(2)

        # item->user
        rsc = i
        dst = u
        h2 = self.gat_layer_1(g, (rsc, dst))
        h2['user'] = h2['user'].squeeze(2)

        # item influence embedding: item->user
        rsc = h1
        dst = u
        item_influence_embedding = self.gat_layer_2(g, (rsc, dst))
        item_influence_embedding['user'] = item_influence_embedding['user'].squeeze(1)

        # social item embedding: user->user
        rsc = h2
        dst = u
        social_item_embedding = self.gat_layer_2(g, (rsc, dst))
        social_item_embedding['user'] = social_item_embedding['user'].squeeze(1)


        # print(item_influence_embedding, social_item_embedding)
        output = torch.cat([item_influence_embedding['user'], social_item_embedding['user']], dim=1)
        return self.output(output)

class SpectralAttentionLayer(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1, kernel_nums=3):
        super(SpectralAttentionLayer, self).__init__()
        self.spec_gcn = dglnn.ChebConv(embedding_size, embedding_size, kernel_nums)
        self.att = dglnn.GATv2Conv(embedding_size, embedding_size, num_heads, allow_zero_in_degree=True)

    def forward(self, social_networks, user_embed, laplacian_lambda_max):
        # spectral gcn
        # print(u)
        user_embed = self.spec_gcn(social_networks, user_embed, laplacian_lambda_max)
        user_embed = self.spec_gcn(social_networks, user_embed, laplacian_lambda_max)

        # attention
        user_embed = self.att(social_networks, user_embed)
        return user_embed.squeeze(1)

class MutualisicLayer(nn.Module):
    def __init__(self, embedding_size=1):
        super(MutualisicLayer, self).__init__()
        self.consumption_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )

        self.social_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )

    def forward(self, raw_embed, consumption_pref, social_pref):
        # print(raw_embed)
        # print(consumption_pref)
        # print(social_pref)
        h_uP = self.consumption_mlp(torch.hstack([consumption_pref, raw_embed]))
        h_uS = self.social_mlp(torch.hstack([social_pref, raw_embed]))

        h_m = h_uP * h_uS

        atten_P = torch.softmax(h_uP, dim=1)
        h_mP = h_m * atten_P
        # h_mP = torch.matmul(atten_P, h_m)   # n * d
        atten_S = torch.softmax(h_uS, dim=1)
        h_mS = h_m * atten_S
        # h_mS = torch.matmul(atten_S, h_m)

        h_mP = torch.hstack([h_mP, h_uP])
        h_mS = torch.hstack([h_mS, h_uS])
        return h_mP, h_mS

class PredictionLayer(nn.Module):
    def __init__(self, embedding_size=1):
        super(PredictionLayer, self).__init__()
        self.mutual_pref_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )
        self.mutual_social_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )


    def forward(self, **inputs):
        h_miu_mP = inputs['h_miu_mP']
        h_miu_mS = inputs['h_miu_mS']

        h_new_P = self.mutual_pref_mlp(h_miu_mP)
        h_new_S = self.mutual_social_mlp(h_miu_mS)

        return h_new_P, h_new_S

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            if etype == 'rate':
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score']
            else:
                n_f_u = h['user']
                i_f = h['item']
                r_f_u = h['raw_user']
                graph.ndata['n_f'] = {'user': n_f_u, 'item': i_f}
                graph.ndata['r_f'] = {'user': r_f_u, 'item': i_f}
                graph.apply_edges(fn.u_dot_v('n_f', 'r_f', 'score'), etype=etype)
                return graph.edges[etype].data['score']

class MutualRecModel(nn.Module):
    def __init__(self, config, rel_names):
        super(MutualRecModel, self).__init__()
        embedding_size = eval(config['MODEL']['embedding_size'])
        num_heads = eval(config['MODEL']['num_heads'])
        num_kernels = eval(config['MODEL']['num_kernels'])
        num_nodes = {
            'user': eval(config['MODEL']['total_user_num']),
            'item': eval(config['MODEL']['item_num'])
        }
        self.embedding = dglnn.HeteroEmbedding(num_nodes, embedding_size)
        self.embedding_user_BN = nn.BatchNorm1d(embedding_size)
        self.embedding_item_BN = nn.BatchNorm1d(embedding_size)

        self.spatial_atten_layer = SpatialAttentionLayer(rel_names, embedding_size, num_heads)

        self.spectral_atten_layer = SpectralAttentionLayer(embedding_size, num_heads, num_kernels)

        self.mutualistic_layer = MutualisicLayer(embedding_size)
        self.prediction_layer = PredictionLayer(embedding_size)

        self.pred = HeteroDotProductPredictor()

    def forward(self, train_pos_g, train_neg_rate_g, train_neg_link_g, social_network, laplacian_lambda_max):
        idx = {ntype: train_pos_g.nodes(ntype) for ntype in train_pos_g.ntypes}
        user_item_embed = self.embedding(idx)
        # item_embed = self.embedding({'item': g.nodes('item')})
        # print(user_item_embed)
        user_embed = self.embedding_user_BN(user_item_embed['user'])
        item_embed = self.embedding_item_BN(user_item_embed['item'])

        user_item_embed = self.spatial_atten_layer(train_pos_g, user_embed, item_embed)
        user_social_embed = self.spectral_atten_layer(social_network, user_embed, laplacian_lambda_max)

        h_miu_mP, h_miu_mS = self.mutualistic_layer(user_embed, user_item_embed, user_social_embed)

        h_new_P, h_new_S = self.prediction_layer(
            h_miu_mP = h_miu_mP,
            h_miu_mS = h_miu_mS,
        )
        res_P_embed = {
            'user': h_new_P,
            'item': item_embed
        }
        pos_rate_score = self.pred(train_pos_g, res_P_embed, 'rate')
        neg_rate_score = self.pred(train_neg_rate_g, res_P_embed, 'rate')

        res_S_embed = {
            'user': h_new_S,
            'item': item_embed,
            'raw_user': user_embed
        }
        pos_link_score = self.pred(train_pos_g, res_S_embed, 'trust')
        neg_link_score = self.pred(train_neg_link_g, res_S_embed, 'trust')
        return pos_rate_score, neg_rate_score, pos_link_score, neg_link_score

    def predict(self, message_g, pos_pred_g, neg_pred_rate_g, neg_pred_link_g, social_network, laplacian_lambda_max):
        idx = {ntype: message_g.nodes(ntype) for ntype in message_g.ntypes}
        user_item_embed = self.embedding(idx)
        # item_embed = self.embedding({'item': g.nodes('item')})
        # print(user_item_embed)
        user_embed = self.embedding_user_BN(user_item_embed['user'])
        item_embed = self.embedding_item_BN(user_item_embed['item'])

        user_item_embed = self.spatial_atten_layer(message_g, user_embed, item_embed)
        user_social_embed = self.spectral_atten_layer(social_network, user_embed, laplacian_lambda_max)

        h_miu_mP, h_miu_mS = self.mutualistic_layer(user_embed, user_item_embed, user_social_embed)

        h_new_P, h_new_S = self.prediction_layer(
            h_miu_mP=h_miu_mP,
            h_miu_mS=h_miu_mS,
        )
        res_P_embed = {
            'user': h_new_P,
            'item': item_embed
        }
        pos_rate_score = self.pred(pos_pred_g, res_P_embed, 'rate')
        neg_rate_score = self.pred(neg_pred_rate_g, res_P_embed, 'rate')

        res_S_embed = {
            'user': h_new_S,
            'item': item_embed,
            'raw_user': user_embed
        }
        pos_link_score = self.pred(pos_pred_g, res_S_embed, 'trust')
        neg_link_score = self.pred(neg_pred_link_g, res_S_embed, 'trust')
        return pos_rate_score, neg_rate_score, pos_link_score, neg_link_score


if __name__ == '__main__':
    pred_user = [0, 1, 2, 3]
    total_user = [0, 1, 2, 3, 4, 5]
    item = [6, 7, 8]
    u = torch.tensor([0, 0, 1, 1, 2, 3, 0, 0, 1, 1, 2, 3, ])
    v = torch.tensor([6, 7, 8, 6, 7, 8, 1, 2, 3, 4, 5, 4, ])
    g = dgl.graph((u, v), num_nodes=9)

    model = MutualRecModel()
    graphs = get_graphs_for_spatial_att(g, 1, 4, 6)
    social_neighbour_network, laplacian_lambda_max = get_social_neighbour_network(g, 4, 6, 1)
    graphs['social_neighbour_network'] = social_neighbour_network
    print(model(graphs, laplacian_lambda_max))
