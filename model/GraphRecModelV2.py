#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   GraphRecModelV2.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/21 18:07   zxx      1.0         None
"""
"""
参考：https://github.com/xhcgit/GraphRec-implicit
"""

# import lib
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch
import torch.nn as nn

from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)


    def forward(self, src_emb, dst_emb):
        x = torch.cat((src_emb, dst_emb), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        scores = self.att3(x)
        return scores


class UserAgg(nn.Module):
    def __init__(self, trainMat, user_num, item_num, rating_num, embedding_size, device, act=None):
        super(UserAgg, self).__init__()
        self.user_num, self.item_num = user_num, item_num
        self.uv_mat = trainMat

        self.embedding_size = embedding_size
        self.shape = torch.Size(self.uv_mat.shape)
        self.act = act

        # item_idx, user_idx
        row_idxs, col_idxs = self.uv_mat.nonzero()
        self.uv_g = dgl.graph(data=(row_idxs, col_idxs + self.user_num),
                              idtype=torch.int32,
                              num_nodes=self.user_num + self.item_num,
                              device=device)

        self.row_idxs = torch.LongTensor(row_idxs).cuda()
        self.col_idxs = torch.LongTensor(col_idxs).cuda()
        self.rating = torch.from_numpy(self.uv_mat.data).long().cuda()
        self.idxs = torch.from_numpy(np.vstack((row_idxs, col_idxs)).astype(np.int64)).cuda()

        # self.w_r1 = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.gu = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        self.att = Attention(self.embedding_size)
        self.w = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, graph, user_feat, item_feat, rating_feat):
        rating = graph.edges['rate'].data['rating']
        u, i = graph.edges(etype='rate')
        r_emb = rating_feat(rating)
        u_emb = user_feat[u]
        i_emb = item_feat[i]

        # original peper formula (15)
        x = torch.cat([u_emb, r_emb], dim=1)
        f_jt = self.gu(x)

        # f_jt = F.relu(self.w_r1(torch.cat([u_emb, r_emb], dim=1)))
        weight = self.att(f_jt, i_emb).view(-1, 1)
        value = edge_softmax(self.uv_g, weight)

        self.uv_g.edata['h'] = f_jt * value

        self.uv_g.update_all(message_func=fn.copy_edge(edge='h', out='m'),
                             reduce_func=fn.sum(msg='m', out='n_f'))

        z = self.uv_g.ndata['n_f'][self.user_num:]

        if self.act is None:
            z = self.w(z)
        else:
            z = self.act(self.w(z))
        return z


class SocialAgg(nn.Module):
    def __init__(self, user_num, trustMat, embedding_size, device, act):
        super(SocialAgg, self).__init__()
        self.user_num = user_num
        self.uu_mat = trustMat
        self.embedding_size = embedding_size
        self.shape = torch.Size(trustMat.shape)
        self.act = act

        row_idxs, col_idxs = self.uu_mat.nonzero()

        self.uu_g = dgl.graph(data=(row_idxs, col_idxs),
                              idtype=torch.int32,
                              num_nodes=self.user_num,
                              device=device)

        self.uu_g.add_self_loop()

        self.row_idxs = torch.LongTensor(row_idxs).cuda()
        self.col_idxs = torch.LongTensor(col_idxs).cuda()
        self.idxs = torch.from_numpy(np.vstack((row_idxs, col_idxs)).astype(np.int64)).cuda()

        self.att = Attention(self.embedding_size)
        self.w = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, uu_g, user_feat, hi):
        trust, trustee = uu_g.edges()
        trust_emb = user_feat[trust]

        trustee_emb = hi[trustee]

        weight = self.att(trust_emb, trustee_emb).view(-1, 1)

        # value = edge_softmax(self.uu_g, weight, norm_by='src').view(-1)
        value = edge_softmax(uu_g, weight).view(-1)



        A = torch.sparse.FloatTensor(self.idxs, value, self.shape).detach()
        A = A.transpose(0, 1)

        if self.act is None:
            hs = self.w(torch.spmm(A, hi))
        else:
            hs = self.act(self.w(torch.spmm(A, hi)))
        return hs


class ItemAgg(nn.Module):
    def __init__(self, user_num, item_num, embedding_size, act=None):
        super(ItemAgg, self).__init__()
        self.user_num, self.item_num = user_num, item_num
        self.embedding_size = embedding_size
        self.shape = torch.Size(self.vu_mat.shape)
        self.act = act

        # # item_idx, user_idx
        # row_idxs, col_idxs = self.vu_mat.nonzero()
        # self.vu_g = dgl.graph(data=(row_idxs + self.user_num, col_idxs),
        #                       idtype=torch.int32,
        #                       num_nodes=self.user_num + self.item_num,
        #                       device=device)
        #
        # self.row_idxs = torch.LongTensor(row_idxs).cuda()
        # self.col_idxs = torch.LongTensor(col_idxs).cuda()
        # self.rating = torch.from_numpy(self.vu_mat.data).long().cuda()
        # self.idxs = torch.from_numpy(np.vstack((row_idxs, col_idxs)).astype(np.int64)).cuda()

        # self.w_r1 = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.gv = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU()
        )
        self.att = Attention(self.embedding_size)
        self.w = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, vu_g, user_feat, item_feat, rating_feat):
        with vu_g.local_scope():
            i, u = vu_g.edges() # graph的rated-by子图
            r = vu_g.edata['rating']
            r_emb = rating_feat[r]
            i_emb = item_feat[i]
            u_emb = user_feat[u]

            # original peper formula (2)
            x = torch.cat([i_emb, r_emb], dim=1)
            x_ia = self.gv(x)

            weight = self.att(x_ia, u_emb).view(-1, 1)
            value = edge_softmax(vu_g, weight)

            vu_g.edata['h'] = x_ia * value

            vu_g.update_all(message_func=fn.copy_edge(edge='h', out='m'),
                                 reduce_func=fn.sum(msg='m', out='n_f'))

            h = self.vu_g.ndata['n_f'][:self.user_num]

            if self.act is None:
                hi = self.w(h)
            else:
                hi = self.act(self.w(h))

            return hi


class GraphRec(nn.Module):
    def __init__(self, config, embedding_size, trainMat, trustMat, device):
        super(GraphRec, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.num_heads = eval(config['MODEL']['num_heads'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.rating_num = eval(config['MODEL']['rating_num'])
        self.drop_rate = eval(config['MODEL']['drop_rate'])
        self.neg_num = eval(config['DATA']['neg_num'])

        self.trainMat = trainMat
        self.trustMat = trustMat

        if args.act == 'relu':
            self.act = nn.ReLU()
        elif args.act == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = None
        self.act = nn.LeakyReLU(0.2)

        # self.user_num, self.item_num = trainMat.shape
        # self.embedding_size = embedding_size
        # self.ratingClass = np.unique(trainMat.data).size

        initializer = nn.init.xavier_uniform_
        # init embedding
        self.userEmbedding = nn.Parameter(initializer(torch.empty(self.user_num, embedding_size)))
        self.itemEmbedding = nn.Parameter(initializer(torch.empty(self.item_num, embedding_size)))
        self.ratingEmbedding = nn.Parameter(initializer(torch.empty(self.rating_num, embedding_size)))


        self.itemAgg = ItemAgg(self.user_num, self.item_num, self.embedding_size, self.act)
        self.socialAgg = SocialAgg(self.trustMat, self.hide_dim, device, self.act)
        self.W2 = nn.Linear(self.hide_dim * 2, self.hide_dim)
        self.userAgg = UserAgg(self.trainMat, self.hide_dim, device, self.act)

    def forward(self):
        hI = self.itemAgg(self.userEmbedding, self.itemEmbedding, self.ratingEmbedding)
        hS = self.socialAgg(self.userEmbedding, hI)
        # original paper formula 12-14
        h = self.W2(torch.cat([hI, hS], dim=1))
        if self.act is not None:
            h = self.act(h)

        z = self.userAgg(self.userEmbedding, self.itemEmbedding, self.ratingEmbedding)

        return h, z