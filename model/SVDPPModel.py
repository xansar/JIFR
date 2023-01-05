#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   TrustSVDModel.py
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
    def forward(self, graph, etype, h, b=None):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            if b is not None:
                graph.ndata['b'] = b
                graph.apply_edges(fn.e_add_u('score', 'b', 'score'), etype=etype)
                graph.apply_edges(fn.v_add_e('b', 'score', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class SVDPPModel(nn.Module):
    def __init__(self, config, rel_names):
        super(SVDPPModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.lamda = eval(config['OPTIM']['lamda'])
        self.lamda_t = eval(config['OPTIM']['lamda_t'])
        global_bias = eval(config['MODEL']['global_bias'])

        self.p_q_embedding = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        self.y_w_embedding = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        self.bias = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, 1
        )
        self.y_gcn = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='left', weight=False, bias=False)
            for rel in rel_names
        })
        self.w_gcn = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='left', weight=False, bias=False)
            for rel in rel_names
        })
        self.u_bias = nn.Embedding(self.user_num, 1)
        self.i_bias = nn.Embedding(self.item_num, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_bias), requires_grad=False)
        self.pred = HeteroDotProductPredictor()

        self.reg_loss = RegLoss(lamda=self.lamda, lamda_t=self.lamda_t)

    def forward(self, message_g, pos_pred_g, neg_pred_g):
        # etype: rate, rated-by, trusted-by
        idx = {ntype: message_g.nodes(ntype) for ntype in message_g.ntypes}
        y_w = self.y_w_embedding(idx)  # {'user': w, 'item': y}
        p_q = self.p_q_embedding(idx)  # {'user': p, 'item': q}
        normed_y = self.y_gcn(message_g,
                              ({'item': y_w['item']}, {'user': p_q['user']}))  # 'user': user_num * embedsize
        normed_w = self.w_gcn(message_g,
                              ({'user': y_w['user']}, {'user': p_q['user']}))  # 'user': user_num * embedsize
        bias = self.bias(idx)
        res_embedding = {'user': normed_y['user'] + p_q['user'], 'item': p_q['item']}

        pos_score = self.pred(pos_pred_g, 'rate', res_embedding, bias) + self.global_bias
        neg_score = self.pred(neg_pred_g, 'rate', res_embedding, bias) + self.global_bias

        # 正则化
        ## social link reg
        with message_g.local_scope():
            message_g.nodes['user'].data['p'] = p_q['user']
            message_g.nodes['user'].data['w'] = y_w['user']
            # 这里是v trusted-by u，所以前面的节点特征用w，后面的用p
            message_g.apply_edges(fn.u_dot_v('w', 'p', 'score'), etype='trusted-by')
            link_pred = message_g.edges['trusted-by'].data['score']

        params = {
            'bias': bias,
            'y_w': y_w,
            'p_q': p_q
        }
        reg_loss= self.reg_loss(message_g, params, link_pred)
        return pos_score, neg_score, reg_loss

class RegLoss(nn.Module):
    def __init__(self, lamda=0.5, lamda_t=0.45):
        super(RegLoss, self).__init__()
        self.lamda = lamda
        self.lamda_t = lamda_t
        self.link_mse = nn.MSELoss()

    def forward(self, graph, params, link_pred):

        # 正则化项
        bias = params['bias']
        p_q = params['p_q']
        y_w = params['y_w']

        I_u_factor = graph.out_degrees(etype='rate')   # user_num
        reg_b_u = torch.mean(self.lamda * I_u_factor * torch.norm(bias['user']))

        U_j_factor = graph.out_degrees(etype='rated-by')   # item_num
        reg_b_j = torch.mean(self.lamda * U_j_factor * torch.norm(bias['item']))

        ## in_degrees+trusted-by——表示当前用户相信的人的数量
        T_u_factor = graph.in_degrees(etype='trusted-by')   # user_num
        reg_p_u = torch.mean(self.lamda * I_u_factor* torch.norm(p_q['user']))

        reg_q_j = torch.mean(self.lamda * U_j_factor * torch.norm(p_q['item']))

        reg_y_i = torch.mean(self.lamda * U_j_factor * torch.norm(y_w['item']))

        return reg_b_u + reg_b_j + reg_p_u + reg_q_j + reg_y_i