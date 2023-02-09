#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/12/3 14:03   zxx      1.0         None
"""
import numpy as np
from numba.typed import List as nList
from numba import njit
from numba.types import ListType, int64, int32, Array

import torch
import dgl

from typing import Dict, List as Dict, List
# import lib
# 使用numba加速负采样
# 测试
@njit(Array(int64, 1, 'C')(ListType(int64,), ListType(ListType(int64,),), int64, int64))
def total_neg_sampling(u_lst, pos_lst, neg_num, total_num):
    lst = []
    lst.append(0)
    lst.pop()
    for u in u_lst:
        history = pos_lst[u]
        for _ in range(neg_num):
            j = np.random.randint(total_num)
            while j in history:
                j = np.random.randint(total_num)
            lst.append(j)
    return np.array(lst)

# 训练子图负采样
@njit(Array(int64, 1, 'C')(ListType(int64,), ListType(ListType(int64,),), Array(int64, 1, 'C'), int64))
def subg_neg_sampling(u_lst, pos_lst, map_array, neg_num):
    lst = []
    lst.append(0)
    lst.pop()
    total_num = map_array.shape[0]
    for u in u_lst:
        history = pos_lst[u]
        relabeled_history_lst = []
        for v in history:
            if v not in map_array:
                continue
            else:
                relabeled_history_lst.append(np.where(map_array == v)[0][0])
        for _ in range(neg_num):
            j = np.random.randint(total_num)
            while j in relabeled_history_lst:
                j = np.random.randint(total_num)
            lst.append(j)
    return np.array(lst)

class SAINTSamplerForHetero(dgl.dataloading.SAINTSampler):
    def node_sampler(self, g):
        """Node ID sampler for random node sampler"""
        # Alternatively, this can be realized by uniformly sampling an edge subset,
        # and then take the src node of the sampled edges. However, the number of edges
        # is typically much larger than the number of nodes.

        sampled_nodes = {ntype: torch.empty(0, dtype=g.idtype, device=g.device) for ntype in g.ntypes}
        if self.prob is None:
            self.prob = {e_t: None for e_t in g.etypes}
        for etype in g.canonical_etypes:
            src_ntype, etype, dst_ntype = etype
            if self.cache and self.prob[etype] is not None:
                prob = self.prob[etype]
            else:
                # 采的src节点
                prob = g.out_degrees(etype=etype).float().clamp(min=1)
                if self.cache:
                    self.prob[etype] = prob

            sampled_srcs = torch.multinomial(prob, num_samples=self.budget,
                                  replacement=True).unique().type(g.idtype)
            sampled_nodes[src_ntype] = torch.cat([sampled_nodes[src_ntype], sampled_srcs]).unique().type(g.idtype)

        return sampled_nodes

    def edge_sampler(self, g):
        """Node ID sampler for random edge sampler"""
        sampled_nodes = {ntype: torch.empty(0, dtype=g.idtype, device=g.device) for ntype in g.ntypes}
        if self.prob is None:
            self.prob = {e_t: None for e_t in g.etypes}
        for etype in g.canonical_etypes:
            src_ntype, etype, dst_ntype = etype
            src, dst = g.edges(etype=etype)
            if self.cache and self.prob[etype] is not None:
                prob = self.prob[etype]
            else:
                in_deg = g.in_degrees(etype=etype).float().clamp(min=1)
                out_deg = g.out_degrees(etype=etype).float().clamp(min=1)
                # We can reduce the sample space by half if graphs are always symmetric.
                prob = 1. / in_deg[dst.long()] + 1. / out_deg[src.long()]
                prob /= prob.sum()
                if self.cache:
                    self.prob[etype] = prob
            sampled_edges = torch.unique(dgl.random.choice(len(prob), size=self.budget, prob=prob))
            if src_ntype == dst_ntype:
                sampled_nodes[src_ntype] = torch.cat([sampled_nodes[src_ntype], src[sampled_edges], dst[sampled_edges]]).unique().type(g.idtype)
            else:
                sampled_nodes[src_ntype] = torch.cat([sampled_nodes[src_ntype], src[sampled_edges]]).unique().type(g.idtype)
                sampled_nodes[dst_ntype] = torch.cat([sampled_nodes[dst_ntype], dst[sampled_edges]]).unique().type(g.idtype)

        return sampled_nodes

    def sample(self, g, indices):
        """Sampling function

        Parameters
        ----------
        g : DGLGraph
            The graph to sample from.
        indices : Tensor
            Placeholder not used.

        Returns
        -------
        DGLGraph
            The sampled subgraph.
        """
        node_ids = self.sampler(g)
        sg = g.subgraph(node_ids, relabel_nodes=True, output_device=self.output_device)
        dgl.dataloading.base.set_node_lazy_features(sg, self.prefetch_ndata)
        dgl.dataloading.base.set_edge_lazy_features(sg, self.prefetch_edata)
        return sg

class NegativeSampler(object):
    def __init__(self, history_lst, total_num, k):
        self.history_lst: Dict[etype: str, etype_history_lst: List[int]] = history_lst
        self.total_num: Dict[etype: str, etype_total_num: int] = total_num
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            _, edge_type, _ = etype
            src, _ = g.find_edges(eids, etype=etype)
            neg_src = src.repeat_interleave(self.k)

            unique_u, idx_in_u_lst = torch.unique(src, return_inverse=True)
            u_lst = nList(unique_u.tolist())
            neg_samples = torch.from_numpy(
                total_neg_sampling(u_lst, self.history_lst[edge_type], self.k, self.total_num[edge_type])).reshape(-1, self.k)
            neg_dst = neg_samples[idx_in_u_lst].reshape(-1)
            result_dict[etype] = (neg_src.to(g.device), neg_dst.to(g.device))
        return result_dict

# 早停用
class EarlyStopper:
    def __init__(self, patience=10, minimum_impro=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.minimum_impro = minimum_impro  # 每次的最小提升

    def __call__(self, indicator_metric, **kwargs):
        if self.best_score is None:
            self.best_score = indicator_metric
        elif indicator_metric < self.best_score + self.minimum_impro:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = indicator_metric
            self.counter = 0

# 画图
def draw_bins_metric_3D_bar_fig(info_table=None):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    #
    # ax = plt.figure().add_subplot(projection='3d')
    # # X, Y, Z = axes3d.get_test_data(0.05)
    # x = range(8)
    # y = range(8)
    # X, Y = np.meshgrid(x, y)
    # # Z = np.random.rand(5, 5) * 0.3
    # Z = np.array([
    #         [0.0203, 0.0329, 0.0296, 0.0278, 0.0260, 0.0194, 0.0678, 0.0000],
    #         [0.0254, 0.0333, 0.0232, 0.0372, 0.0229, 0.0580, 0.0000, 0.0538],
    #         [0.0295, 0.0200, 0.0375, 0.0192, 0.0354, 0.0221, 0.0275, 0.0741],
    #         [0.0425, 0.0336, 0.0403, 0.0366, 0.0389, 0.0489, 0.0307, 0.0745],
    #         [0.0350, 0.0190, 0.0309, 0.0303, 0.0365, 0.0356, 0.0540, 0.0428],
    #         [0.0464, 0.0395, 0.0372, 0.0471, 0.0467, 0.0478, 0.0375, 0.0463],
    #         [0.0487, 0.0369, 0.0478, 0.0538, 0.0541, 0.0490, 0.0528, 0.0454],
    #         [0.0477, 0.0806, 0.0763, 0.1041, 0.0678, 0.0661, 0.0709, 0.0606],
    # ])

    # setup the figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    x = np.arange(8)
    y = np.arange(8)
    X, Y = np.meshgrid(x, y)
    X, Y = X.ravel(), Y.ravel()
    Z = np.array([
            [0.0203, 0.0329, 0.0296, 0.0278, 0.0260, 0.0194, 0.0678, 0.0000],
            [0.0254, 0.0333, 0.0232, 0.0372, 0.0229, 0.0580, 0.0000, 0.0538],
            [0.0295, 0.0200, 0.0375, 0.0192, 0.0354, 0.0221, 0.0275, 0.0741],
            [0.0425, 0.0336, 0.0403, 0.0366, 0.0389, 0.0489, 0.0307, 0.0745],
            [0.0350, 0.0190, 0.0309, 0.0303, 0.0365, 0.0356, 0.0540, 0.0428],
            [0.0464, 0.0395, 0.0372, 0.0471, 0.0467, 0.0478, 0.0375, 0.0463],
            [0.0487, 0.0369, 0.0478, 0.0538, 0.0541, 0.0490, 0.0528, 0.0454],
            [0.0477, 0.0806, 0.0763, 0.1041, 0.0678, 0.0661, 0.0709, 0.0606],
    ])
    top = Z.ravel()
    bottom = np.zeros_like(top)
    width = depth = 1

    # # fake data
    # _x = np.arange(4)
    # _y = np.arange(5)
    # _xx, _yy = np.meshgrid(_x, _y)
    # x, y = _xx.ravel(), _yy.ravel()
    #
    # top = x + y
    # bottom = np.zeros_like(top)
    # width = depth = 1


    ax1.bar3d(X, Y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    ax2.bar3d(X, Y, bottom, width, depth, top, shade=False)
    ax2.set_title('Not Shaded')

    plt.show()

if __name__ == '__main__':
    draw_bins_metric_3D_bar_fig()