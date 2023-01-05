#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   NJBPModel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/12/14 14:01   zxx      1.0         None
"""

# import lib

import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl.function as fn

from .utils import init_weights

class NeuralInterestNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(NeuralInterestNetwork, self).__init__()
        self.embedding_size = embedding_size

        ## interest
        self.interest_mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU(),
        )

        self.output_hL = nn.Linear(self.embedding_size, 1)
        self.output_h1 = nn.Linear(self.embedding_size, 1)
        self.output_h2 = nn.Linear(self.embedding_size, 1)

    def forward(self, graph, embed_linears):
        with graph.local_scope():
            # deep
            ## 这里把论文公式5的拼接后再变换的操作变成先分别变换再相加
            graph.nodes['user'].data.update({'c1': embed_linears['s'] + embed_linears['p']})
            graph.nodes['item'].data.update({'c2': embed_linears['q1'] + embed_linears['q2']})
            graph.apply_edges(fn.u_add_v('c1', 'c2', 'c'), etype='rate')
            c = graph.edges['rate'].data.pop('c')
            h_L = self.interest_mlp(c)

            # shallow
            graph.nodes['user'].data.update({'x': embed_linears['x']})
            graph.nodes['item'].data.update({'m': embed_linears['m']})
            graph.apply_edges(fn.u_mul_v('x', 'm', 'h1'), etype='rate')
            h1 = graph.edges['rate'].data.pop('h1')

            graph.nodes['user'].data.update({'w': embed_linears['w']})
            graph.apply_edges(fn.u_mul_v('w', 'm', 'h2'), etype='rate')
            h2 = graph.edges['rate'].data.pop('h2')
            output = self.output_hL(h_L) + self.output_h1(h1) + self.output_h2(h2)
            return output

class NeuralTrustNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(NeuralTrustNetwork, self).__init__()
        self.embedding_size = embedding_size

        ## trust
        self.trust_mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU(),
        )

        self.output_hL = nn.Linear(self.embedding_size, 1)
        self.output_h1 = nn.Linear(self.embedding_size, 1)
        self.output_h2 = nn.Linear(self.embedding_size, 1)

    def forward(self, graph, embed_linears):
        with graph.local_scope():
            # deep
            graph.nodes['user'].data.update({'s': embed_linears['s1'] + embed_linears['s2']})
            graph.nodes['user'].data.update({'p': embed_linears['p1'] + embed_linears['p2']})
            graph.apply_edges(fn.u_add_v('s', 'p', 'c'), etype='trust')
            c = graph.edges['trust'].data.pop('c')
            h_L = self.trust_mlp(c)

            # shallow
            graph.nodes['user'].data.update({'x': embed_linears['x']})
            graph.apply_edges(fn.u_mul_v('x', 'x', 'h1'), etype='trust')
            h1 = graph.edges['trust'].data.pop('h1')

            graph.nodes['user'].data.update({'w': embed_linears['w']})
            graph.apply_edges(fn.u_mul_v('w', 'w', 'h2'), etype='trust')
            h2 = graph.edges['trust'].data.pop('h2')

            output = self.output_hL(h_L) + self.output_h1(h1) + self.output_h2(h2)
            return output

class NJBPModel(nn.Module):
    def __init__(self, config, rel_names):
        super(NJBPModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])

        ## P&Q
        self.deep_consumption_embed = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        ## W&M
        self.shallow_consumption_embed = dglnn.HeteroEmbedding(
            {'user': self.user_num, 'item': self.item_num}, self.embedding_size
        )
        ## S&X
        self.social_embed = dglnn.HeteroEmbedding(
            {'deep': self.user_num, 'shallow': self.user_num}, self.embedding_size
        )

        self.interest_embed_linears = dglnn.HeteroLinear({
            k: self.embedding_size
            for k in ['s', 'p', 'q1', 'q2']
        }, self.embedding_size)
        self.trust_embed_linears = dglnn.HeteroLinear({
            k: self.embedding_size
            for k in ['s1', 's2', 'p1', 'p2']
        }, self.embedding_size)


        ## trust
        ## interest
        self.interest_net = NeuralInterestNetwork(self.embedding_size)
        self.trust_net = NeuralTrustNetwork(self.embedding_size)

        init_weights(self.modules())

    def forward(self, message_g, pos_pred_g, neg_pred_g, input_nodes=None):
        if input_nodes is None:
            if '_ID' in message_g.ndata.keys():
                # 子图采样的情况
                input_nodes = {ntype: message_g.nodes[ntype].data['_ID'] for ntype in message_g.ntypes}
            else:
                # 全图的情况
                input_nodes = {ntype: message_g.nodes(ntype=ntype) for ntype in message_g.ntypes}
        else:
            message_g = message_g[0]
            # 这里没有卷积操作，直接使用dstnodes
            dst_nodes = {ntype: message_g.dstnodes(ntype) for ntype in message_g.dsttypes}
            input_nodes = {ntype: idx[dst_nodes[ntype]] for ntype, idx in input_nodes.items()}
        p_q = self.deep_consumption_embed(input_nodes)
        w_m = self.shallow_consumption_embed(input_nodes)
        s_x = self.social_embed({
            'shallow': input_nodes['user'],
            'deep': input_nodes['user'],
        })
        ## deep
        interest_embed_linears = self.interest_embed_linears({
            's': s_x['deep'],
            'p': p_q['user'],
            'q1': p_q['item'],
            'q2': p_q['item'],
        })
        ## shallow
        interest_embed_linears.update({
            'x': s_x['shallow'],
            'w': w_m['user'],
            'm': w_m['item'],
        })
        # interest
        pos_interest_score = self.interest_net(pos_pred_g, interest_embed_linears)
        neg_interest_score = self.interest_net(neg_pred_g, interest_embed_linears)
        ## deep
        trust_embed_linears = self.trust_embed_linears({
            's1': s_x['deep'],
            's2': s_x['deep'],
            'p1': p_q['user'],
            'p2': p_q['user'],
        })
        ## shallow
        trust_embed_linears.update({
            'x': s_x['shallow'],
            'w': w_m['user'],
        })
        # trust
        pos_trust_score = self.trust_net(pos_pred_g, trust_embed_linears)
        neg_trust_score = self.trust_net(neg_pred_g, trust_embed_linears)
        return pos_interest_score, neg_interest_score, pos_trust_score, neg_trust_score




if __name__ == '__main__':
    from utils import init_weights
    import dgl
    u = torch.randint(0, 10, (30,))
    i = torch.randint(0, 20, (30,))
    v = torch.randint(0, 10, (30,))
    graph_data = {
        ('user', 'rate', 'item'): (u, i),
        ('item', 'rated-by', 'user'): (i, u),
        ('user', 'trust', 'user'): (u, v),
        ('user', 'trusted-by', 'user'): (v, u),
    }
    num_nodes = {
        'user': 10,
        'item': 20
    }
    g = dgl.heterograph(graph_data, num_nodes)
    graph_data = {
        ('user', 'rate', 'item'): (u, i),
        ('user', 'trust', 'user'): (u, v),
    }
    neg_g = dgl.heterograph(graph_data, num_nodes)

    model = NJBPModel()
    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())
    loss_func = nn.BCEWithLogitsLoss()
    for i in range(1000):
        optimizer.zero_grad()
        output = model(g, g, neg_g)
        # loss = loss_func(output[0], torch.ones_like(output[0])) + \
        #        loss_func(output[1], torch.zeros_like(output[1])) + \
        #        loss_func(output[2], torch.ones_like(output[2])) + \
        #        loss_func(output[3], torch.zeros_like(output[3]))
        loss = loss_func(output[0], torch.ones_like(output[0])) + \
               loss_func(output[2], torch.ones_like(output[2]))
        loss.backward()
        optimizer.step()
        print(loss.item())