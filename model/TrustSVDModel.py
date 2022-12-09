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

from .utils import init_weights


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


class TrustSVDModel(nn.Module):
    def __init__(self, config, rel_names):
        super(TrustSVDModel, self).__init__()
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
            rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='none', weight=False, bias=False)
            for rel in rel_names
        })
        self.w_gcn = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(self.embedding_size, self.embedding_size, norm='none', weight=False, bias=False)
            for rel in rel_names
        })
        self.global_bias = nn.Parameter(torch.tensor(global_bias), requires_grad=True)
        self.pred = HeteroDotProductPredictor()
        init_weights(self.modules())

        self.reg_loss = RegLoss(lamda=self.lamda, lamda_t=self.lamda_t)

    def forward(self, messege_g, pos_pred_g, neg_pred_g, input_nodes=None):
        if input_nodes is None:
            if '_ID' in messege_g.ndata.keys():
                # 子图采样的情况
                input_nodes = {ntype: messege_g.nodes[ntype].data['_ID'] for ntype in messege_g.ntypes}
            else:
                # 全图的情况
                input_nodes = {ntype: messege_g.nodes(ntype=ntype) for ntype in messege_g.ntypes}
        else:
            messege_g = messege_g[0]
        y_w = self.y_w_embedding(input_nodes)  # {'user': w, 'item': y}
        p_q = self.p_q_embedding(input_nodes)  # {'user': p, 'item': q}

        src_user = messege_g.srcnodes(ntype='user')
        src_item = messege_g.srcnodes(ntype='item')
        dst_user = messege_g.dstnodes(ntype='user')
        dst_item = messege_g.dstnodes(ntype='item')

        I_u_mask = (messege_g.out_degrees(etype='rate') > 0).float()[dst_user]
        I_u_factor = (I_u_mask / torch.sqrt(messege_g.out_degrees(etype='rate').clamp(min=1))[dst_user]).reshape(-1, 1)
        normed_y = I_u_factor * self.y_gcn(messege_g,
                                           ({'item': y_w['item'][src_item]}, {'user': y_w['user'][dst_user]}))[
            'user']  # 'user': user_num * embedsize

        T_u_mask = (messege_g.in_degrees(etype='trusted-by') > 0).float()[dst_user]
        T_u_factor = (T_u_mask / torch.sqrt(messege_g.in_degrees(etype='trusted-by').clamp(min=1))[dst_user]).reshape(
            -1, 1)
        normed_w = T_u_factor * self.w_gcn(messege_g,
                                           ({'user': y_w['user'][src_user]}, {'user': y_w['user'][dst_user]}))[
            'user']  # 'user': user_num * embedsize

        bias = self.bias({'user': dst_user, 'item': dst_item})
        res_embedding = {
            'user': normed_w + normed_y + p_q['user'][dst_user],
            'item': p_q['item'][dst_item]
        }

        pos_score = self.pred(pos_pred_g, 'rate', res_embedding, bias) + self.global_bias
        neg_score = self.pred(neg_pred_g, 'rate', res_embedding, bias) + self.global_bias

        # 正则化
        ## social link reg
        with messege_g.local_scope():

            messege_g.srcnodes['user'].data['w'] = y_w['user']
            messege_g.dstnodes['user'].data['p'] = p_q['user'][messege_g.dstnodes(ntype='user')]
            # 这里是v trusted-by u，所以前面的节点特征用w，后面的用p
            messege_g.apply_edges(fn.u_dot_v('w', 'p', 'score'), etype='trusted-by')
            link_pred = messege_g.edges['trusted-by'].data['score']

        params = {
            'bias': bias,
            'y_w': y_w,
            'p_q': p_q,
            'I_u_factor': I_u_factor,
            'T_u_factor':T_u_factor
        }
        reg_loss, link_loss = self.reg_loss(messege_g, params, link_pred)
        return pos_score, neg_score, reg_loss, link_loss
    # deprecated forward
        # if input_nodes is None:
        # # etype: rate, rated-by, trusted-by
        #     idx = {ntype: messege_g.nodes[ntype].data['_ID'] for ntype in messege_g.ntypes}
        #     # full graph: idx = {ntype: messege_g.nodes(ntype) for ntype in messege_g.ntypes}
        #     y_w = self.y_w_embedding(idx)  # {'user': w, 'item': y}
        #     p_q = self.p_q_embedding(idx)  # {'user': p, 'item': q}
        #     I_u_mask = (messege_g.out_degrees(etype='rate') > 0).float()
        #     I_u_factor = (I_u_mask / torch.sqrt(messege_g.out_degrees(etype='rate').clamp(min=1))).reshape(-1, 1)
        #     normed_y = I_u_factor * self.y_gcn(messege_g,
        #                           ({'item': y_w['item']}, {'user': p_q['user']}))['user']  # 'user': user_num * embedsize
        #
        #     T_u_mask = (messege_g.in_degrees(etype='trusted-by') > 0).float()
        #     T_u_factor = (T_u_mask / torch.sqrt(messege_g.in_degrees(etype='trusted-by').clamp(min=1))).reshape(-1, 1)
        #     normed_w = T_u_factor * self.w_gcn(messege_g,
        #                               ({'user': y_w['user']}, {'user': p_q['user']}))['user']  # 'user': user_num * embedsize
        #
        #     bias = self.bias(idx)
        #     res_embedding = {
        #         'user': normed_w + normed_y + p_q['user'],
        #         'item': p_q['item']
        #     }
        #
        # else:
        #     y_w = self.y_w_embedding(input_nodes)  # {'user': w, 'item': y}
        #     p_q = self.p_q_embedding(input_nodes)  # {'user': p, 'item': q}
        #     messege_g = messege_g[0]
        #     src_user = messege_g.srcnodes(ntype='user')
        #     src_item = messege_g.srcnodes(ntype='item')
        #     dst_user = messege_g.dstnodes(ntype='user')
        #     dst_item = messege_g.dstnodes(ntype='item')
        #     I_u_mask = (messege_g.out_degrees(etype='rate') > 0).float()[dst_user]
        #     I_u_factor = (I_u_mask / torch.sqrt(messege_g.out_degrees(etype='rate').clamp(min=1))[dst_user]).reshape(-1, 1)
        #     normed_y = I_u_factor * self.y_gcn(messege_g,
        #                           ({'item': y_w['item'][src_item]}, {'user': y_w['user'][dst_user]}))['user']  # 'user': user_num * embedsize
        #
        #     T_u_mask = (messege_g.in_degrees(etype='trusted-by') > 0).float()[dst_user]
        #     T_u_factor = (T_u_mask / torch.sqrt(messege_g.in_degrees(etype='trusted-by').clamp(min=1))[dst_user]).reshape(-1, 1)
        #     normed_w = T_u_factor * self.w_gcn(messege_g,
        #                               ({'user': y_w['user'][src_user]}, {'user': y_w['user'][dst_user]}))['user']  # 'user': user_num * embedsize
        #
        #     bias = self.bias({'user': dst_user, 'item': dst_item})
        #     res_embedding = {
        #         'user': normed_w + normed_y + p_q['user'][dst_user],
        #         'item': p_q['item'][dst_item]
        #     }

class RegLoss(nn.Module):
    def __init__(self, lamda=0.5, lamda_t=0.25):
        super(RegLoss, self).__init__()
        self.lamda = lamda
        self.lamda_t = lamda_t
        self.link_mse = nn.MSELoss()

    def forward(self, graph, params, link_pred):
        # link_loss
        link_loss = self.lamda_t * self.link_mse(link_pred, torch.ones_like(link_pred, device=link_pred.device))

        src_user = graph.srcnodes(ntype='user')
        src_item = graph.srcnodes(ntype='item')
        dst_user = graph.dstnodes(ntype='user')
        dst_item = graph.dstnodes(ntype='item')
        # 正则化项
        bias = params['bias']
        p_q = params['p_q']
        y_w = params['y_w']
        I_u_factor = params['I_u_factor'].reshape(-1)
        T_u_factor = params['T_u_factor'].reshape(-1)

        reg_b_u = torch.mean(self.lamda * I_u_factor * torch.sum(torch.square(bias['user'][dst_user]), dim=1))

        U_j_mask = (graph.out_degrees(etype='rated-by') > 0).float()[dst_item]
        U_j_factor = U_j_mask / torch.sqrt(graph.out_degrees(etype='rated-by').clamp(min=1))[dst_item]   # item_num
        reg_b_j = torch.mean(self.lamda * U_j_factor * torch.sum(torch.square(bias['item'][dst_item]), dim=1))

        reg_p_u = torch.mean((self.lamda * I_u_factor + self.lamda_t * T_u_factor) *
                             torch.sum(torch.square(p_q['user'][dst_user]), dim=1))

        reg_q_j = torch.mean(self.lamda * U_j_factor * torch.sum(torch.square(p_q['item'][dst_item]), dim=1))

        reg_y_i = torch.mean(self.lamda * U_j_factor * torch.sum(torch.square(y_w['item'][dst_item]), dim=1))

        ## out_degrees+trusted-by——表示相信当前用户的人的数量
        T_v_p_mask = (graph.out_degrees(etype='trusted-by') > 0).float()[dst_user]
        T_v_p_factor = T_v_p_mask / torch.sqrt(graph.out_degrees(etype='trusted-by').clamp(min=1))[dst_user]   # user_num
        reg_w_v = torch.mean(self.lamda_t * T_v_p_factor * torch.sum(torch.square(y_w['user'][dst_user]), dim=1))

        return reg_b_u + reg_b_j + reg_p_u + reg_q_j + reg_y_i + reg_w_v, link_loss