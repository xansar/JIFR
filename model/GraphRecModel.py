#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   GraphRecModel.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/8 14:59   zxx      1.0         None
"""

# import lib
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
弃用
"""
class GraphRecGAT(dglnn.GATv2Conv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 e_feats=None,
                 drop_rate=0.,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False
    ):
        super(GraphRecGAT, self).__init__(
                 in_feats=in_feats,
                 out_feats=out_feats,
                 num_heads=num_heads,
                 feat_drop=feat_drop,
                 attn_drop=attn_drop,
                 negative_slope=negative_slope,
                 residual=residual,
                 activation=activation,
                 allow_zero_in_degree=allow_zero_in_degree,
                 bias=bias,
                 share_weights=share_weights
        )
        # linear+act+linear，graphrec的注意力计算公式
        self.edge_linear = nn.Linear(out_feats, out_feats)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_linear.weight, gain=gain)

        if e_feats is not None:
            self.feat_e = nn.Linear(e_feats, out_feats * num_heads, bias=bias)
            self.e_act_and_mlp = nn.Sequential(
                # 这里是user/item embedding和opinion embedding线性变换、相加之后需要激活
                # 相当于融合层三层mlp中的第一层
                nn.LeakyReLU(negative_slope),
                # nn.Dropout(feat_drop),
                # 第二层mlp
                nn.Linear(out_feats * num_heads, out_feats * num_heads, bias=bias),
                nn.LeakyReLU(negative_slope),
                # 这里开始进入attention，相当于attention开始先对特征做线性变换
                nn.Linear(out_feats * num_heads, out_feats * num_heads, bias=bias),
            )


    def forward(self, graph, feat, e_feat=None, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            """
            GraphRec的aggregation需要处理item/user embedding和opinion embedding
            的融合，这里把opinion看作边的特征，因此在进行节点注意力计算前需要先更新source侧节点特征
            具体地，先对节点和边特征分别进行一次线性变换，然后相加，这一操作等价于拼接后线性变换。
            然后将变换后的节点特征与边特征相加，过一次leakyrelu
            论文里是三层mlp，因此还需将边特征送入两层mlp内
            """
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
            if e_feat is not None:
                h_e = self.feat_drop(e_feat)
                feat_e = self.feat_e(h_e)
                graph.edata.update({'e': feat_e})

            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})

            if e_feat is not None:
                graph.apply_edges(fn.u_add_e('el', 'e', 'e'))
                graph.edata.update({'e': self.e_act_and_mlp(graph.edata['e'])})
                graph.apply_edges(fn.e_add_v('e', 'er', 'e'))
            else:
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
            # print(e)
            """
            GraphRec在计算注意力时，使用了linear+act+linear的模式
            所以要加一个
            """
            e = self.edge_linear(e)
            # print(e)

            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)
            # message passing
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class AttentionWithEpinions(nn.Module):
    def __init__(self, embedding_size, dropout_rate=0.5):
        super(AttentionWithEpinions, self).__init__()
        self.src_linear = nn.Linear(embedding_size, embedding_size)
        self.dst_linear = nn.Linear(embedding_size, embedding_size)

        self.final_mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(embedding_size, 1),
        )
    def forward(self, graph, src_feat_with_epinions, dst_feat):
        with graph.local_scope():
            # 计算注意力，相当于边和dst节点计算注意力，三层mlp
            ## 第一层，对e和dst特征分别线性变换并加
            e_feat = self.src_linear(src_feat_with_epinions)
            right_feat = self.dst_linear(dst_feat)
            graph.edata['e_ft'] = e_feat
            graph.dstdata.update({'r_ft': right_feat})
            graph.apply_edges(fn.e_add_v('e_ft', 'r_ft', 'score'))
            ## 激活，后接dropout与激活
            graph.edata['score'] = self.final_mlp(graph.edata['score'])
            graph.edata['score'] = edge_softmax(graph, graph.edata['score'].view(-1, 1))
            return graph.edata['score']


class AggregationWithEpinions(nn.Module):
    def __init__(self, embedding_size, allow_zero_in_degree=True, dropout_rate=0.5):
        super(AggregationWithEpinions, self).__init__()
        self.allow_zero_in_degree = allow_zero_in_degree
        self.dropout = nn.Dropout(dropout_rate)

        self.src_linear = nn.Linear(embedding_size, embedding_size)
        self.dst_linear = nn.Linear(embedding_size, embedding_size)
        self.epinions_linear = nn.Linear(embedding_size, embedding_size)
        self.conbined_feat_linear = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU()
        )
        self.att = AttentionWithEpinions(embedding_size, dropout_rate)
        self.output = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU()
        )
    def forward(self, graph, feat, epinions):
        with graph.local_scope():
            if not self.allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                # 先对src特征和epinions分别线性变换后加和激活，等价于拼接后变换激活
                ## 变换
                feat_src = self.dropout(self.src_linear(feat[0]))
                feat_dst = self.dropout(self.dst_linear(feat[1]))
                feat_epinions = self.epinions_linear(epinions)
                feat_epinions = self.dropout(feat_epinions[graph.edata['rating']])
                ## 加和
                graph.srcdata.update({'el': feat_src})
                graph.dstdata.update({'er': feat_dst})

                graph.edata['e'] = feat_epinions
                graph.apply_edges(fn.u_add_e('el', 'e', 'e'))
                ## 第一层激活，第二层线性变换+激活，得到带epinion的用户/物品向量，储存在边上
                graph.edata['e'] = self.conbined_feat_linear(F.leaky_relu(graph.edata['e']))

                # 注意力
                ## 经过edge softmax的注意力score
                score = self.att(graph, graph.edata['e'], graph.dstdata['er'])
                ## 加权
                graph.edata['e'] = score * graph.edata['e']
                ## 求和
                graph.update_all(message_func=fn.copy_edge(edge='e', out='m'),
                                     reduce_func=fn.sum(msg='m', out='n_f'))
                ## 激活后返回
                return self.output(graph.dstdata['n_f'])
            else:
                raise ValueError("Please put the nodes feat in tuple(src,dst)!")


class AggregationWithoutEpinions(nn.Module):
    def __init__(self, embedding_size, allow_zero_in_degree=True, dropout_rate=0.5):
        super(AggregationWithoutEpinions, self).__init__()
        self.allow_zero_in_degree = allow_zero_in_degree
        self.dropout = nn.Dropout(dropout_rate)

        self.src_linear = nn.Linear(embedding_size, embedding_size)
        self.dst_linear = nn.Linear(embedding_size, embedding_size)

        self.att = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(embedding_size, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU()
        )

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self.allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                ## 变换
                feat_src = self.dropout(self.src_linear(feat[0]))
                feat_dst = self.dropout(self.dst_linear(feat[1]))
                ## 加和
                graph.srcdata.update({'el': feat_src})
                graph.dstdata.update({'er': feat_dst})
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                ## 注意力
                graph.edata['score'] = self.att(graph.edata['e'])
                graph.edata['score'] = edge_softmax(graph, graph.edata['score'].view(-1, 1))

                graph.edata['e'] = graph.edata['score'] * graph.edata['e']

                ## 求和
                graph.update_all(message_func=fn.copy_edge(edge='e', out='m'),
                                 reduce_func=fn.sum(msg='m', out='n_f'))
                ## 激活后返回
                return self.output(graph.ndata['n_f'])
            else:
                raise ValueError("Please put the nodes feat in tuple(src,dst)!")

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class GraphRecModel(nn.Module):
    def __init__(self, config, rel_names):
        super(GraphRecModel, self).__init__()
        self.config = config
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.num_heads = eval(config['MODEL']['num_heads'])
        self.task = config['TRAIN']['task']
        self.pred_user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.rating_num = eval(config['MODEL']['rating_num'])
        self.drop_rate = eval(config['MODEL']['drop_rate'])
        self.neg_num = eval(config['DATA']['neg_num'])
        self.embedding = dglnn.HeteroEmbedding({
            'user': self.total_user_num,
            'item': self.item_num,
            'rating': self.rating_num
        }, self.embedding_size)

        self.user_social_item_aggregation = dglnn.HeteroGraphConv({
            rel: AggregationWithEpinions(self.embedding_size, True, self.drop_rate)
            if 'rate' in rel else
            AggregationWithoutEpinions(self.embedding_size, True, self.drop_rate)
            for rel in rel_names
        })

        self.user_latent_factor_mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            nn.LeakyReLU()
        )

        self.pred = HeteroDotProductPredictor()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, message_g, pos_pred_g, neg_pred_g):
        idx = {ntype: message_g.nodes(ntype) for ntype in message_g.ntypes}
        idx.update({'rating': torch.arange(6, device=idx['user'].device)})
        embedding = self.embedding(idx)

        # 这里rate和rated-by上rating序号是相同的
        # assert (message_g.edges['rate'].data['rating'] != message_g.edges['rated-by'].data['rating']).sum()==0
        edge_embedding = embedding['rating']

        # user modeling
        ## item aggregation
        item_space = self.user_social_item_aggregation(
            message_g,
            ({'item': embedding['item']}, {'user': embedding['user']}),
            mod_kwargs={'rated-by': {'epinions': edge_embedding}}
        )['user']
        assert item_space.shape == (self.total_user_num, self.embedding_size)

        ## social aggregation
        social_space = self.user_social_item_aggregation(
            message_g,
            ({'user': item_space}, {'user': embedding['user']})
        )['user']
        assert social_space.shape == (self.total_user_num, self.embedding_size)

        ## concation and mlp
        user_latent_factor = self.user_latent_factor_mlp(torch.cat([item_space, social_space], dim=1))
        assert user_latent_factor.shape == (self.total_user_num, self.embedding_size)

        # item modeling
        item_latent_factor = self.user_social_item_aggregation(
            message_g,
            ({'user': embedding['user']}, {'item': embedding['item']}),
            mod_kwargs={'rate': {'epinions': edge_embedding}}
        )['item']
        assert item_latent_factor.shape == (self.item_num, self.embedding_size)

        # prediction
        pos_score = self.pred(pos_pred_g, {'user': user_latent_factor, 'item': item_latent_factor}, 'rate')
        neg_score = self.pred(neg_pred_g, {'user': user_latent_factor, 'item': item_latent_factor}, 'rate')

        assert pos_score.shape[1] == 1
        return pos_score, neg_score