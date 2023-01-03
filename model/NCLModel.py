#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   NCLModel.py
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

from .kmeans import kmeans
from .utils import init_weights

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class NCLModel(nn.Module):
    def __init__(self, config, rel_names):
        super(NCLModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.layer_num = eval(config['MODEL']['gcn_layer_num'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])

        self.ssl_temp = eval(config['MODEL']['ssl_temp'])
        self.ssl_reg = eval(config['MODEL']['ssl_reg'])
        self.alpha = eval(config['MODEL']['alpha'])
        self.proto_reg = eval(config['MODEL']['proto_reg'])
        self.k = eval(config['MODEL']['num_clusters'])

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

    def _compute_single_type_neighbour_ssl_loss(self, current_embedding, previous_embedding, previous_totoal_embedding):
        norm_current_embedding = F.normalize(current_embedding)
        norm_previous_embedding = F.normalize(previous_embedding)
        norm_previous_totoal_embedding = F.normalize(previous_totoal_embedding)
        pos_score = torch.mul(norm_current_embedding, norm_previous_embedding).sum(dim=1)
        total_score = torch.matmul(norm_current_embedding, norm_previous_totoal_embedding.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.ssl_temp) # n
        total_score = torch.exp(total_score / self.ssl_temp).sum(dim=1)# n*N
        ssl_loss = -torch.log(pos_score / total_score).sum()
        return ssl_loss

    def compute_structral_neighbour_ssl_loss(self, used_embeddings, total_embeddings):

        # user
        ssl_user_loss = self._compute_single_type_neighbour_ssl_loss(
            current_embedding=used_embeddings['user'][1],
            previous_embedding=used_embeddings['user'][0],
            previous_totoal_embedding=total_embeddings['user']
        )
        # item
        ssl_item_loss = self._compute_single_type_neighbour_ssl_loss(
            current_embedding=used_embeddings['item'][1],
            previous_embedding=used_embeddings['item'][0],
            previous_totoal_embedding=total_embeddings['item']
        )
        ssl_loss = self.ssl_reg * (ssl_user_loss + self.alpha * ssl_item_loss)
        return ssl_loss

    def e_step(self):
        user_embeddings = self.embedding.weight['user'].detach()
        item_embeddings = self.embedding.weight['item'].detach()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings, 'user')
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings, 'item')

    @torch.no_grad()
    def run_kmeans(self, x, node_type='user'):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        # kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        # kmeans.train(x)
        # cluster_cents = kmeans.centroids
        #
        # _, I = kmeans.index.search(x, 1)

        cluster_ids_x, cluster_centers = kmeans(
            X=x, num_clusters=self.k, distance='euclidean', device=torch.device('cuda:0'),
            node_type=node_type
        )

        # convert to cuda Tensors for broadcast
        centroids = cluster_centers.to(x.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = cluster_ids_x.to(x.device)
        return centroids, node2cluster

    def compute_single_type_protoNCE_loss(self, dtype, embeddings, idx):
        norm_embeddings = F.normalize(embeddings)
        if dtype == 'user':
            node_2cluser = self.user_2cluster
            node_2centroids = self.user_centroids
        else:
            node_2cluser = self.item_2cluster
            node_2centroids = self.item_centroids
        node2cluster = node_2cluser[idx]
        node2centroids = node_2centroids[node2cluster]
        pos_score = torch.mul(norm_embeddings, node2centroids).sum(dim=1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        total_score = torch.matmul(norm_embeddings, node_2centroids.transpose(0, 1))
        total_score = torch.exp(total_score / self.ssl_temp).sum(dim=1)
        proto_nce_loss = -torch.log(pos_score / total_score).sum()
        return proto_nce_loss

    def compute_protoNCE_loss(self, user_embeddings, item_embeddings, user, item):
        proto_nce_user_loss = self.compute_single_type_protoNCE_loss('user', user_embeddings, user)
        proto_nce_item_loss = self.compute_single_type_protoNCE_loss('item', item_embeddings, item)
        proto_nce_loss = self.proto_reg * (proto_nce_user_loss + proto_nce_item_loss)
        return proto_nce_loss

    def forward(self, messege_g, pos_pred_g, neg_pred_g, input_nodes=None):
        if input_nodes is None:
            if '_ID' in messege_g.ndata.keys():
                # 子图采样的情况
                idx = {ntype: messege_g.nodes[ntype].data['_ID'] for ntype in messege_g.ntypes}
            else:
                # 全图的情况
                idx = {ntype: messege_g.nodes(ntype=ntype) for ntype in messege_g.ntypes}
            res_embeddings = self.embedding(idx)
            used_embeddings = {
                'user': [res_embeddings['user'], ], # 0是初始embedding，1是第k（2）层的embedding
                'item': [res_embeddings['item'], ]
            }
            for i, layer in enumerate(self.layers):
                if i == 0:
                    embeddings = layer(messege_g, res_embeddings)
                else:
                    embeddings = layer(messege_g, embeddings)

                res_embeddings['user'] = res_embeddings['user'] + embeddings['user']
                res_embeddings['item'] = res_embeddings['item'] + embeddings['item']

            # 这里简化了，因为只有两层gcn，所以取最后一层embeddings就是结构邻居
            used_embeddings['user'].append(embeddings['user'])
            used_embeddings['item'].append(embeddings['item'])
            total_embeddings = self.embedding.weight
            ssl_loss = self.compute_structral_neighbour_ssl_loss(used_embeddings, total_embeddings)

            # 语义邻居损失
            proto_nce_loss = self.compute_protoNCE_loss(
                user_embeddings=self.embedding.weight['user'][idx['user']], # 取出用到的部分
                item_embeddings=self.embedding.weight['item'][idx['item']],
                user=idx['user'],
                item=idx['item']
            )

            res_embeddings['user'] /= len(self.layers) + 1
            res_embeddings['item'] /= len(self.layers) + 1
        else:
            original_embedding = self.embedding(input_nodes)
            dst_user = pos_pred_g.dstnodes(ntype='user')
            dst_item = pos_pred_g.dstnodes(ntype='item')
            res_embeddings = {
                'user': original_embedding['user'][dst_user],
                'item': original_embedding['item'][dst_item]
            }
            # used_embeddings = {
            #     'user': [res_embeddings['user'], ], # 0是初始embedding，1是第k（2）层的embedding
            #     'item': [res_embeddings['item'], ]
            # }
            for i in range(1, len(messege_g) + 1):
                blocks = messege_g[-i:]
                for j in range(i):
                    layer = self.layers[j]
                    embed = {
                        'user': original_embedding['user'][blocks[j].srcnodes('user')],
                        'item': original_embedding['item'][blocks[j].srcnodes('item')],
                    }
                    if j == 0:
                        # embeddings = layer(blocks[j], original_embedding)
                        embeddings = layer(blocks[j], embed)
                    else:
                        embeddings = layer(blocks[j], embeddings)
                # # ssl loss
                # if i == len(messege_g):
                #     used_embeddings['user'].append(embeddings['user'])
                #     used_embeddings['item'].append(embeddings['item'])

                res_embeddings['user'] = res_embeddings['user'] + embeddings['user']
                res_embeddings['item'] = res_embeddings['item'] + embeddings['item']

            # total_embeddings = self.embedding.weight
            # ssl_loss = self.compute_structral_neighbour_ssl_loss(used_embeddings, total_embeddings)
            #
            # proto_nce_loss = self.compute_protoNCE_loss(
            #     user_embeddings=self.embedding.weight['user'][dst_user],
            #     item_embeddings=self.embedding.weight['item'][dst_item],
            #     user=dst_user,
            #     item=dst_item
            # )
            ssl_loss, proto_nce_loss = \
                torch.tensor(torch.nan, device=pos_pred_g.device), torch.tensor(torch.nan, device=pos_pred_g.device)
            res_embeddings['user'] /= len(self.layers) + 1
            res_embeddings['item'] /= len(self.layers) + 1
        pos_score = self.pred(pos_pred_g, res_embeddings, 'rate')
        neg_score = self.pred(neg_pred_g, res_embeddings, 'rate')

        return pos_score, neg_score, ssl_loss, proto_nce_loss