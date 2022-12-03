#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Node2VecModel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/28 14:27   zxx      1.0         None
"""

# import lib
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import dgl.function as fn

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

## 参考：https://github.com/dmlc/dgl/tree/master/examples/pytorch/node2vec
class Node2VecModel(nn.Module):
    """Node2vec model from paper node2vec: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.  Same notation as in the paper.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        Same notation as in the paper.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, use PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.
        If omitted, DGL assumes that the neighbors are picked uniformly.
    """

    def __init__(self, config):
        super(Node2VecModel, self).__init__()

        self.config = config
        self.user_num = eval(config['MODEL']['user_num'])
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.p = eval(config['MODEL']['p'])
        self.q = eval(config['MODEL']['q'])
        self.num_walks = eval(config['MODEL']['num_walks'])
        self.walk_length = eval(config['MODEL']['walk_length'])
        self.window_size = eval(config['MODEL']['window_size'])
        assert self.walk_length >= self.window_size
        self.num_negatives = eval(config['DATA']['train_neg_num'])
        # if weight_name is not None:
        #     self.prob = weight_name
        # else:
        #     self.prob = None

        self.embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.pred = HeteroDotProductPredictor()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.
        Returns
        -------
        Tensor
            Node embedding
        """
        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]

    def loss(self, pos_trace, neg_trace):
        """
        Computes the loss given positive and negative random walks.
        Parameters
        ----------
        pos_trace: Tensor
            positive random walk trace
        neg_trace: Tensor
            negative random walk trace
        """
        e = 1e-15

        # Positive
        pos_start, pos_rest = (
            pos_trace[:, 0],
            pos_trace[:, 1:].contiguous(),
        )  # start node and following trace
        w_start = self.embedding(pos_start).unsqueeze(dim=1)
        w_rest = self.embedding(pos_rest)
        pos_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # Negative
        neg_start, neg_rest = neg_trace[:, 0], neg_trace[:, 1:].contiguous()

        w_start = self.embedding(neg_start).unsqueeze(dim=1)
        w_rest = self.embedding(neg_rest)
        neg_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # compute loss
        pos_loss = -torch.log(torch.sigmoid(pos_out) + e).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + e).mean()

        return pos_loss, neg_loss

    def evaluate(self, message_g, pos_pred_g, neg_pred_g):
        """
        Evaluate the quality of embedding vector via a downstream classification task with logistic regression.
        """
        res_embedding = self.embedding(message_g.nodes(ntype='user'))
        pos_score = self.pred(pos_pred_g, res_embedding, 'trust')
        neg_score = self.pred(neg_pred_g, res_embedding, 'trust')

        return pos_score, neg_score