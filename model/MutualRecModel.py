#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MutualRecModel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/28 18:41   zxx      1.0         None
"""

# import lib
import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

class ItemInfluenceEmbedding(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1):
        self.embedding_size = embedding_size
        super(ItemInfluenceEmbedding, self).__init__()
        self.user2item_gat = dglnn.GATv2Conv(
            in_feats=embedding_size,
            out_feats=embedding_size,
            num_heads=num_heads,
            activation=torch.nn.functional.leaky_relu,
            allow_zero_in_degree=True
        )
        self.item_influence_gat = dglnn.GATv2Conv(
            in_feats=embedding_size,
            out_feats=embedding_size,
            num_heads=num_heads,
            activation=torch.nn.functional.leaky_relu,
            allow_zero_in_degree=True
        )

    def forward(self, user2item_g, reverse_consumption_neighbour_g, embedding):
        # 应该只有被采样的item部分向量不为0
        user2item_embedding = self.user2item_gat(user2item_g, embedding).reshape(-1, self.embedding_size)    # num_nodes * embedding_size
        # print(user2item_embedding)
        # 只会使用上面不为0的item，其他的都不会使用，返回的是pred user部分不为0
        item_influence_embedding = self.item_influence_gat(reverse_consumption_neighbour_g, user2item_embedding).reshape(-1, self.embedding_size)
        # print(item_influence_embedding)
        return item_influence_embedding


class SocialItemEmbedding(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1):
        self.embedding_size = embedding_size
        super(SocialItemEmbedding, self).__init__()
        self.item2user_gat = dglnn.GATv2Conv(
            in_feats=embedding_size,
            out_feats=embedding_size,
            num_heads=num_heads,
            activation=torch.nn.functional.leaky_relu,
            allow_zero_in_degree=True
        )
        self.social_item_gat = dglnn.GATv2Conv(
            in_feats=embedding_size,
            out_feats=embedding_size,
            num_heads=num_heads,
            activation=torch.nn.functional.leaky_relu,
            allow_zero_in_degree=True
        )

    def forward(self, item2user_g, social_neighbour_g, embedding):
        # 取出采样图用到的embedding
        social_used_embedding = embedding[item2user_g.ndata['_ID']]
        # 应该只有被采样的user部分向量不为0
        item2user_embedding = self.item2user_gat(item2user_g, social_used_embedding).reshape(-1, self.embedding_size)    # num_nodes * embedding_size
        # 只会使用上面不为0的user，其他的都不会使用，返回的是pred user部分不为0

        # 只包括pred user的嵌入
        social_embedding = embedding[social_neighbour_g.nodes()]    # 这里只用到采样节点的embedding
        # 找到进行user-user信息传输的起始user节点id，并在social_embedding中修改为item2user_embedding
        used_user_id = torch.where(torch.sum(item2user_embedding, dim=1) != 0)[0]
        social_user_id = item2user_g.ndata['_ID'][used_user_id] # 对应consumption_g的id，也对应social_neighbour_g的id
        # social_user_id：使用的这几个用户在social图中的id，used_user_id：这几个用户在item2user_g中的id
        social_embedding[social_user_id] = item2user_embedding[used_user_id]
        social_item_embedding = self.social_item_gat(social_neighbour_g, social_embedding).reshape(-1, self.embedding_size)
        return social_item_embedding

class SpatialAttentionLayer(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1):
        super(SpatialAttentionLayer, self).__init__()
        self.item_influence_embedding = ItemInfluenceEmbedding(embedding_size, num_heads)
        self.social_item_embedding = SocialItemEmbedding(embedding_size, num_heads)
        self.output = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )

    def forward(self, user2item_g, reverse_consumption_neighbour_g, item2user_g, social_neighbour_g, embedding):
        # embedding 是截取过的，pred user 和 item部分的
        # 这个embedding，最后信息传递到user上，因此只需要把pred user部分的embedding取出来就行
        item_influence_embedding = \
            self.item_influence_embedding(user2item_g, reverse_consumption_neighbour_g, embedding)[:social_neighbour_g.num_nodes()]
        social_item_embedding = self.social_item_embedding(item2user_g, social_neighbour_g, embedding)
        # 最后返回所有pred user的embedding
        return self.output(torch.cat([item_influence_embedding, social_item_embedding], dim=1))

def get_graphs_for_spatial_att(g, n_neighbours, pred_user_max, consumption_max):
    # item influence, 因为consumption_g包好所有的pred user和item节点，只需要将这两部分的embedding拼起来即可，不需要对应id
    consumption_g = dgl.edge_subgraph(g, edges=range(consumption_max))  # 要注意节点id的对应
    ## 顶层图
    consumption_neighbour_g = dgl.sampling.sample_neighbors(
        g=consumption_g,
        nodes=range(pred_user_max),  # 对所有的用户都采item的邻居
        fanout=n_neighbours,
        replace=False,
        edge_dir='out'  # 都是user指向item
    )
    reverse_consumption_neighbour_g = dgl.reverse(consumption_neighbour_g)

    ## 底层图
    sampled_item = torch.unique(consumption_neighbour_g.edges()[1]) # consumption_graph的id
    user2item_g = dgl.sampling.sample_neighbors(
                g=consumption_graph,
                nodes=sampled_item,
                fanout=n_neighbours,
                replace=False,
                edge_dir='in'
            )

    # social influence
    ## 从整图中抽取社交网络
    pred_user_social_sub_g = dgl.node_subgraph(g, nodes=range(pred_user_max))
    pred_user_social_sub_g = dgl.to_bidirected(pred_user_social_sub_g)

    social_neighbour_g = dgl.sampling.sample_neighbors(
        g=pred_user_social_sub_g,
        nodes=range(pred_user_max),
        fanout=n_neighbours,
        replace=False,
        edge_dir='in'  # 其实in和out采的边的分布一样，只是边的方向不一样
        # 使用in，那么边(u, v)中v就是完整的nodes列表
    )
    ## user列表
    sampled_user = torch.unique(social_neighbour_g.edges()[0]).tolist()
    ## item列表
    item_id_range = list(range(pred_user_max, consumption_g.num_nodes()))
    social_consumption_neighbour_g = dgl.sampling.sample_neighbors(
        g=consumption_g,    # pred user 部分id与social neigh一样，item部分多出来的就是item_id_range的范围
        nodes=range(pred_user_max),  # 对所有的用户都采item的邻居
        fanout=n_neighbours,
        replace=False,
        edge_dir='out'  # 都是user指向item
    )
    # 这里又重新编号了
    item2user_g = dgl.node_subgraph(social_consumption_neighbour_g, nodes=sampled_user + item_id_range)
    item2user_g = dgl.reverse(item2user_g)
    return user2item_g, reverse_consumption_neighbour_g, item2user_g, social_neighbour_g