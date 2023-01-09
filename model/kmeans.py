#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   kmeans.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/12/22 15:58   zxx      1.0         None
"""

# 使用kmeans_pytorch，这里为了修改进度条所以放到model内
# import lib
import gc

import numpy as np
import torch
from tqdm import tqdm

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    # 随机抽取样本作为初始簇中心
    initial_state = X[indices]
    return initial_state

# def initialize(X, num_clusters, distance, device):
#     """
#     initialize cluster centers
#     :param X: (torch.tensor) matrix
#     :param num_clusters: (int) number of clusters
#     :return: (np.array) initial state
#     """
#     if distance == 'euclidean':
#         pairwise_distance_function = pairwise_distance
#     elif distance == 'cosine':
#         pairwise_distance_function = pairwise_cosine
#     else:
#         raise NotImplementedError
#
#     num_samples = len(X)
#
#     first = np.random.choice(num_samples)
#     # 储存在一个列表中
#     indices = [first]
#     # 继续选取k-1个点
#     for i in range(1, num_clusters):
#         selected_centers = X[indices]
#         dis = pairwise_distance_function(X, selected_centers, device=device).reshape(num_samples, -1)
#         min_dis, _ = torch.min(dis, dim=1)
#         p = (min_dis / min_dis.sum()).cpu().numpy()
#         assert p.shape[0] == num_samples
#         idx = np.random.choice(num_samples, p=p)
#         assert idx not in indices
#         indices.append(idx)
#
#
#     # indices = np.random.choice(num_samples, num_clusters, replace=False)
#     # 随机抽取样本作为初始簇中心
#     initial_state = X[indices]
#     return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-2,
        niter=100,
        device=torch.device('cpu'),
        node_type='user',
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running {node_type} k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # batch_size = 4096

    # convert to float
    X = X.float()

    # transfer to device
    # original_X = X.to(device)
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]', total=niter)
    while True:
        # # 这里使用mini batch，就是随机从X中进行抽样，然后基于抽取的样本迭代簇中心
        # num_samples = len(original_X)
        # sampled_idx = np.random.choice(num_samples, batch_size, replace=False)
        # X = original_X[sampled_idx]
        # if iteration == 0:
        #     # initialize
        #     initial_state = initialize(X, num_clusters, distance, device)

        dis = pairwise_distance_function(X, initial_state, device=device)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
            if len(selected.shape) != 0 and selected.shape[0] == 0:
                continue
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol or iteration >= niter:
            break

    # choice_cluster = kmeans_predict(original_X, initial_state, device=device)
    return choice_cluster, initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    batch_size = 16384
    batch_num = int(data1.shape[0] / batch_size) + 1
    B = data2.unsqueeze(dim=0)
    dis = torch.zeros(data1.shape[0], data2.shape[0])
    for i in range(batch_num):
        s = batch_size * i
        e = batch_size * (i + 1) if i != batch_num - 1 else data1.shape[0]
        a = data1[s: e, :].unsqueeze(dim=1)
        assert a.shape[0] <= batch_size and a.shape[1] == 1 and a.shape[2] == data1.shape[1]
        # bsz*K*D
        batch_dis = (a - B) ** 2.0
        assert batch_dis.shape[0] <= batch_size and batch_dis.shape[1] == data2.shape[0] and a.shape[2] == data1.shape[1]
        # bsz*K
        batch_dis = batch_dis.sum(dim=-1).squeeze()
        assert batch_dis.shape[0] <= batch_size and batch_dis.shape[1] == data2.shape[0]
        dis[s: e, :] = batch_dis
        del a, batch_dis
        gc.collect()

    # # N*1*D
    # A = data1.unsqueeze(dim=1)
    #
    # # 1*K*D
    # B = data2.unsqueeze(dim=0)
    #
    # # N*K*D
    # dis = (A - B) ** 2.0
    # # return N*N matrix for pairwise distance
    # # N*K
    # dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

