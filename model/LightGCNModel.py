#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LightGCNModel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/25 18:43   zxx      1.0         None
"""

# import lib
import torch.nn as nn
import torch
import dgl
import dgl.function as fn

class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            # rate: [user, item], link: total_user
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
    def forward(self, graph: dgl.DGLHeteroGraph, node_f):
        with graph.local_scope():
            # D^-1/2
            degs = graph.out_degrees().to(node_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(message_func=fn.copy_src(src='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(node_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            return rst


# 参考：https://github.com/xhcgit/LightGCN-implicit-DGL/blob/master/LightGCN.py
class LightGCNModel(nn.Module):
    def __init__(self, config=None):
        super(LightGCNModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.layer_num = eval(config['MODEL']['layer_num'])
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])

        if self.task == 'Rate':
            self.nodes_num = self.user_num + self.item_num
        elif self.task == 'Link':
            self.nodes_num = self.total_user_num
        self.embedding = nn.Parameter(torch.randn(self.nodes_num, self.embedding_size))

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(GCNLayer())

        self.predictor = DotProductPredictor()

    def forward(self, pos_graph, neg_graph):
        embedding = self.get_embedding(pos_graph)
        pos_pred = self.predictor(pos_graph, embedding)
        neg_pred = self.predictor(neg_graph, embedding)

        return pos_pred.reshape(-1), neg_pred.reshape(-1)

    def get_embedding(self, graph):
        res_embedding = self.embedding

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, res_embedding)
            else:
                embeddings = layer(graph, embeddings)
            res_embedding = res_embedding + embeddings * (1 / (i + 2))

        # user_embedding = res_user_embedding  # / (len(self.layers)+1)
        # item_embedding = res_item_embedding  # / (len(self.layers)+1)
        return res_embedding

if __name__ == '__main__':
    user = torch.tensor([0, 0, 1, 2])
    item = torch.tensor([3, 4, 4, 3])
    rate = torch.tensor([3, 2, 4, 5], dtype=torch.float)
    graph = dgl.graph((user, item))
    model = LightGCNModel()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(lr=1e-1, params=model.parameters(), weight_decay=1e-4)
    for i in range(1000):
        optimizer.zero_grad()
        e = model.get_embedding(graph)
        pred = model.predictor(graph, e)
        loss = loss_func(pred.reshape(-1), rate)
        print(f'{i}: {loss.item()}')
        loss.backward()
        optimizer.step()
