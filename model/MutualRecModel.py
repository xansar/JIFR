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


"""
Spatial att layer
"""

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

def get_graphs_for_spatial_att(g, num_sample, pred_user_size, consumption_size):
    # item influence, 因为consumption_g包好所有的pred user和item节点，只需要将这两部分的embedding拼起来即可，不需要对应id
    consumption_g = dgl.edge_subgraph(g, edges=range(consumption_size))  # 要注意节点id的对应
    ## 顶层图
    consumption_neighbour_g = dgl.sampling.sample_neighbors(
        g=consumption_g,
        nodes=range(pred_user_size),  # 对所有的用户都采item的邻居
        fanout=num_sample,
        replace=False,
        edge_dir='out'  # 都是user指向item
    )
    reverse_consumption_neighbour_g = dgl.reverse(consumption_neighbour_g)

    ## 底层图
    sampled_item = torch.unique(consumption_neighbour_g.edges()[1]) # consumption_graph的id
    user2item_g = dgl.sampling.sample_neighbors(
                g=consumption_g,
                nodes=sampled_item,
                fanout=num_sample,
                replace=False,
                edge_dir='in'
            )

    # social influence
    ## 从整图中抽取社交网络
    pred_user_social_sub_g = dgl.node_subgraph(g, nodes=range(pred_user_size))

    social_neighbour_g = dgl.sampling.sample_neighbors(
        g=pred_user_social_sub_g,
        nodes=range(pred_user_size),
        fanout=num_sample,
        replace=False,
        edge_dir='in'  # 其实in和out采的边的分布一样，只是边的方向不一样
        # 使用in，那么边(u, v)中v就是完整的nodes列表
    )
    social_neighbour_g = dgl.to_bidirected(social_neighbour_g)
    ## user列表
    sampled_user = torch.unique(social_neighbour_g.edges()[0]).tolist()
    ## item列表
    item_id_range = list(range(pred_user_size, consumption_g.num_nodes()))
    social_consumption_neighbour_g = dgl.sampling.sample_neighbors(
        g=consumption_g,    # pred user 部分id与social neigh一样，item部分多出来的就是item_id_range的范围
        nodes=range(pred_user_size),  # 对所有的用户都采item的邻居
        fanout=num_sample,
        replace=False,
        edge_dir='out'  # 都是user指向item
    )
    # 这里又重新编号了
    item2user_g = dgl.node_subgraph(social_consumption_neighbour_g, nodes=sampled_user + item_id_range)
    item2user_g = dgl.reverse(item2user_g)
    graphs = {
        'g': g,
        'user2item_g': user2item_g,
        'reverse_consumption_neighbour_g': reverse_consumption_neighbour_g,
        'item2user_g': item2user_g,
        'social_neighbour_g': social_neighbour_g,
    }
    return graphs

"""
Spectral att layer
"""
def get_social_neighbour_network(g, pred_user_size, total_user_size, num_sample):
    pred_user = range(pred_user_size)
    total_user = range(total_user_size)
    # 采样社交邻接网络，同时计算拉普拉斯矩阵特征值
    ## 采样
    social_network = dgl.node_subgraph(g, total_user)
    social_neighbout_nerwork = dgl.sampling.sample_neighbors(
        social_network,
        pred_user,
        fanout=num_sample,
        replace=False,
        edge_dir='out')
    social_neighbout_nerwork = dgl.to_bidirected(social_neighbout_nerwork)
    ## 计算特征值
    laplacian_lambda_max = dgl.laplacian_lambda_max(social_neighbout_nerwork)
    laplacian_lambda_max = torch.tensor(laplacian_lambda_max, dtype=torch.float32)
    return social_neighbout_nerwork, laplacian_lambda_max


class SpectralAttentionLayer(nn.Module):
    def __init__(self, embedding_size=1, num_heads=1, kernel_nums=3):
        self.embedding_size = embedding_size
        super(SpectralAttentionLayer, self).__init__()
        self.spec_gcn = dglnn.ChebConv(
            embedding_size,
            embedding_size,
            kernel_nums,
            activation=nn.functional.leaky_relu
        )
        self.att = dglnn.GATv2Conv(
            embedding_size,
            embedding_size,
            num_heads,
            allow_zero_in_degree=True,
            activation=nn.functional.leaky_relu
        )

    def forward(self, social_neighbour_network, embedding, laplacian_lambda_max):
        # social network应该从total user中采样，并且是bidirected
        # spectral gcn
        # print(u)
        h = self.spec_gcn(social_neighbour_network, embedding, laplacian_lambda_max).reshape(-1, self.embedding_size)
        h = self.spec_gcn(social_neighbour_network, h, laplacian_lambda_max).reshape(-1, self.embedding_size)

        # attention
        social_preference_embedding = self.att(social_neighbour_network, h).reshape(-1, self.embedding_size)
        return social_preference_embedding

"""
Mutualistic Layer
"""
class MutualisicLayer(nn.Module):
    def __init__(self, embedding_size=1):
        super(MutualisicLayer, self).__init__()
        self.consumption_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )

        self.social_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )

    def forward(self, user_embed, consumption_pref, social_pref):
        # print(raw_embed)
        # print(consumption_pref)
        # print(social_pref)
        h_uP = self.consumption_mlp(torch.hstack([consumption_pref, user_embed]))
        h_uS = self.social_mlp(torch.hstack([social_pref, user_embed]))

        h_m = h_uP * h_uS

        atten_P = torch.softmax(h_uP, dim=1)
        h_mP = h_m * atten_P
        # h_mP = torch.matmul(atten_P, h_m)   # n * d
        atten_S = torch.softmax(h_uS, dim=1)
        h_mS = h_m * atten_S
        # h_mS = torch.matmul(atten_S, h_m)

        h_mP = torch.hstack([h_mP, h_uP])
        h_mS = torch.hstack([h_mS, h_uS])
        return h_mP, h_mS


class PredictionLayer(nn.Module):
    def __init__(self, embedding_size=1):
        super(PredictionLayer, self).__init__()
        self.mutual_pref_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )
        self.mutual_social_mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU()
        )


    def forward(self, h_miu_mP, h_miu_mS):
        h_new_P = self.mutual_pref_mlp(h_miu_mP)
        h_new_S = self.mutual_social_mlp(h_miu_mS)

        return h_new_P, h_new_S

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, embedding):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['ft'] = embedding
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class MutualRecModel(nn.Module):
    def __init__(self, config=None):
        super(MutualRecModel, self).__init__()
        # self.embedding_size = eval(config['MODEL']['embedding_size'])
        # self.num_heads = eval(config['MODEL']['num_heads'])
        # self.num_kernels = eval(config['MODEL']['num_kernels'])
        self.pred_user_num = 4
        self.total_user_num = 6
        self.item_num = 3
        self.embedding_size = 1
        self.num_heads = 1
        self.num_kernels = 3
        self.nodes_num = self.total_user_num + self.item_num
        self.embedding = nn.Embedding(self.nodes_num, self.embedding_size)
        self.embedding_BN = nn.BatchNorm1d(self.embedding_size)

        self.spatial_atten_layer = SpatialAttentionLayer(self.embedding_size, self.num_heads)

        self.spectral_atten_layer = SpectralAttentionLayer(self.embedding_size, self.num_heads, self.num_kernels)

        self.mutualistic_layer = MutualisicLayer(self.embedding_size)
        self.prediction_layer = PredictionLayer(self.embedding_size)

        self.predictor = HeteroDotProductPredictor()

    def forward(self, graphs: dict, laplacian_lambda_max):
        g = graphs['g']
        embedding = self.embedding_BN(self.embedding(g.nodes())) # 总embedding
        pred_user_embedding = embedding[:self.pred_user_num, :]
        total_user_embedding = embedding[:self.total_user_num, :]
        item_embedding = embedding[-self.item_num:, :]
        # Spatial layer
        user2item_g = graphs['user2item_g']
        reverse_consumption_neighbour_g = graphs['reverse_consumption_neighbour_g']
        item2user_g = graphs['item2user_g']
        social_neighbour_g = graphs['social_neighbour_g']
        # 这一层只用pred user部分embedding和item embedding
        user_item_embedding = torch.cat([pred_user_embedding, item_embedding], dim=0)
        user_item_embed = self.spatial_atten_layer(
            user2item_g,
            reverse_consumption_neighbour_g,
            item2user_g,
            social_neighbour_g,
            user_item_embedding
        )

        # Spectral layer
        social_neighbour_network = graphs['social_neighbour_network']
        # 这一层用total user的embedding
        user_social_embed = self.spectral_atten_layer(
            social_neighbour_network,
            total_user_embedding,
            laplacian_lambda_max
        )

        # mutualistic layer
        user_social_embed = user_social_embed[:self.pred_user_num, :]
        h_miu_mP, h_miu_mS = self.mutualistic_layer(pred_user_embedding, user_item_embed, user_social_embed)

        h_new_P, h_new_S = self.prediction_layer(
            h_miu_mP = h_miu_mP,
            h_miu_mS = h_miu_mS,
        )
        return h_new_P, h_new_S

if __name__ == '__main__':
    pred_user = [0, 1, 2, 3]
    total_user = [0, 1, 2, 3, 4, 5]
    item = [6, 7, 8]
    u = torch.tensor([0, 0, 1, 1, 2, 3, 0, 0, 1, 1, 2, 3, ])
    v = torch.tensor([6, 7, 8, 6, 7, 8, 1, 2, 3, 4, 5, 4, ])
    g = dgl.graph((u, v), num_nodes=9)

    model = MutualRecModel()
    graphs = get_graphs_for_spatial_att(g, 1, 4, 6)
    social_neighbour_network, laplacian_lambda_max = get_social_neighbour_network(g, 4, 6, 1)
    graphs['social_neighbour_network'] = social_neighbour_network
    print(model(graphs, laplacian_lambda_max))
