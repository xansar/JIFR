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
    def forward(self, graph, u_f, v_f):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            # 这里拼接了，实际上item embedding的idx要比user大
            h = torch.cat([u_f, v_f], dim=0)
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph: dgl.DGLHeteroGraph, u_f, v_f):
        with graph.local_scope():
            node_f = torch.cat([u_f, v_f], dim=0)
            # D^-1/2
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(message_func=fn.copy_src(src='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            return rst


# 参考：https://github.com/xhcgit/LightGCN-implicit-DGL/blob/master/LightGCN.py
class LightGCNModel(nn.Module):
    def __init__(self, config):
        super(LightGCNModel, self).__init__()
        self.config = config
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.layer_num = eval(config['MODEL']['layer_num'])
        self.U = nn.Parameter(torch.randn(self.user_num, self.embedding_size))
        self.I = nn.Parameter(torch.randn(self.item_num, self.embedding_size))

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(GCNLayer())

        self.predictor = DotProductPredictor()

    def forward(self, pos_graph, neg_graph):
        user_embedding, item_embedding = self.get_embedding(pos_graph)

        pos_pred = self.predictor(pos_graph, user_embedding, item_embedding)
        neg_pred = self.predictor(neg_graph, user_embedding, item_embedding)

        return pos_pred.reshape(-1), neg_pred.reshape(-1)

    def get_embedding(self, graph):
        res_user_embedding = self.U
        res_item_embedding = self.I

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, res_user_embedding, res_item_embedding)
            else:
                embeddings = layer(graph, embeddings[: self.user_num], embeddings[self.user_num:])

            res_user_embedding = res_user_embedding + embeddings[: self.user_num] * (1 / (i + 2))
            res_item_embedding = res_item_embedding + embeddings[self.user_num:] * (1 / (i + 2))

        # user_embedding = res_user_embedding  # / (len(self.layers)+1)
        # item_embedding = res_item_embedding  # / (len(self.layers)+1)
        return res_user_embedding, res_item_embedding

if __name__ == '__main__':
    user = torch.tensor([0, 0, 1, 2])
    item = torch.tensor([3, 4, 4, 3])
    rate = torch.tensor([3, 2, 4, 5], dtype=torch.float)
    graph = dgl.graph((user, item))
    model = LightGCNModel()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(lr=1e-1, params=model.parameters(), weight_decay=1e-4)
    for i in range(100):
        optimizer.zero_grad()
        pred = model(graph=graph)
        loss = loss_func(pred, rate)
        print(f'{i}: {loss.item()}')
        loss.backward()
        optimizer.step()
