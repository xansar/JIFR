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

class GraphRecGAT(dglnn.GATv2Conv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 e_feats=None,
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
                nn.Dropout(feat_drop),
                # 第二层mlp
                nn.Linear(out_feats * num_heads, out_feats * num_heads, bias=bias),
                nn.LeakyReLU(negative_slope),
                nn.Dropout(feat_drop),
                # 第三层mlp
                nn.Linear(out_feats * num_heads, out_feats * num_heads, bias=bias),
                nn.LeakyReLU(negative_slope),
                nn.Dropout(feat_drop),
                # 这里开始进入attention，相当于attention开始先对特征做线性变换
                nn.Linear(out_feats * num_heads, out_feats * num_heads, bias=bias),
            )
            for m in self.e_act_and_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=gain)


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
                graph.edata.update({'ee': feat_e})

            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})

            if e_feat is not None:
                graph.apply_edges(fn.u_add_e('el', 'ee', 'ef'))
                graph.edata.update({'ef': self.e_act_and_mlp(graph.edata['ef'])})
                graph.apply_edges(fn.e_add_v('ef', 'er', 'e'))
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

class CatMLP(nn.Module):
    """
    用来计算两个特征cat之后过mlp，先分别线性变换再相加比先拼接再变换效率高
    """
    def __init__(self, embedding_size, p=0.5):
        super(CatMLP, self).__init__()
        self.feat_drop = nn.Dropout(p)
        self.left_mlp = nn.Linear(embedding_size, embedding_size)
        self.right_mlp = nn.Linear(embedding_size, embedding_size)
        self.leaky_relu = nn.LeakyReLU()
        self.res_mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(p)
        )

    def forward(self, left_embed, right_embed):
        l_f = self.left_mlp(self.feat_drop(left_embed))
        r_f = self.right_mlp(self.feat_drop(right_embed))
        f = self.leaky_relu(l_f + r_f)
        f = self.feat_drop(f)
        f = self.res_mlp(f)
        return f

class HeteroMLPPredictor(nn.Module):
    def __init__(self, embedding_size, p=0.5):
        super(HeteroMLPPredictor, self).__init__()
        # 第一层
        self.feat_drop = nn.Dropout(p)
        self.left_mlp = nn.Linear(embedding_size, embedding_size)
        self.right_mlp = nn.Linear(embedding_size, embedding_size)
        self.leaky_relu = nn.LeakyReLU()
        self.pred_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p),
            # 第二层
            nn.Linear(embedding_size, embedding_size),
            nn.LeakyReLU(),
            nn.Dropout(p),
            # 第三层
            nn.Linear(embedding_size, 1),
            nn.LeakyReLU(),
        )
    def forward(self, graph, h, etype):
        with graph.local_scope():
            # 先分别做线性变换
            u_f = self.left_mlp(self.feat_drop(h['user']))
            i_f = self.right_mlp(self.feat_drop(h['item']))
            graph.ndata['h'] = {'user': u_f, 'item': i_f}
            graph.apply_edges(fn.u_add_v('h', 'h', 'score'), etype=etype)

            return self.pred_linear(graph.edges[etype].data['score'])

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

        self.embedding = dglnn.HeteroEmbedding({
            'user': self.total_user_num,
            'item': self.item_num,
            'rating': self.rating_num
        }, self.embedding_size)

        self.user_social_item_aggregation = dglnn.HeteroGraphConv({
            rel: GraphRecGAT(
                self.embedding_size,
                self.embedding_size,
                self.num_heads,
                e_feats=self.embedding_size,
                feat_drop=self.drop_rate,
                attn_drop=self.drop_rate
            )
            if 'rate' in rel else
            GraphRecGAT(
                self.embedding_size,
                self.embedding_size,
                self.num_heads,
                feat_drop=self.drop_rate,
                attn_drop=self.drop_rate
            )
            for rel in rel_names
        })

        self.user_latent_factor_mlp = CatMLP(self.embedding_size, self.drop_rate)
        self.pred = HeteroMLPPredictor(self.embedding_size, self.drop_rate)

    def forward(self, message_g, pos_pred_g, neg_pred_g):
        idx = {ntype: message_g.nodes(ntype) for ntype in message_g.ntypes}
        idx.update({'rating': torch.arange(6, device=idx['user'].device)})
        embedding = self.embedding(idx)
        # 这里rate和rated-by上rating序号是相同的
        assert (message_g.edges['rate'].data['rating'] != message_g.edges['rated-by'].data['rating']).sum()==0

        edge_embedding = embedding['rating'][message_g.edges['rate'].data['rating']]
        # user modeling
        ## item aggregation
        src = {'item': embedding['item']}
        dst = {'user': embedding['user']}
        item_space = self.user_social_item_aggregation(
            message_g,
            (src, dst),
            mod_kwargs={'rated-by': {'e_feat': edge_embedding}}
        )['user'].squeeze(1)
        assert item_space.shape == (self.total_user_num, self.embedding_size)
        ## social aggregation
        src = {'user': item_space}
        dst = {'user': embedding['user']}
        social_space = self.user_social_item_aggregation(
            message_g,
            (src, dst)
        )['user'].squeeze(1)
        assert social_space.shape == (self.total_user_num, self.embedding_size)
        ## concation and mlp
        user_latent_factor = self.user_latent_factor_mlp(item_space, social_space)
        assert user_latent_factor.shape == (self.total_user_num, self.embedding_size)

        # item modeling
        src = {'user': embedding['user']}
        dst = {'item': embedding['item']}
        item_latent_factor = self.user_social_item_aggregation(
            message_g,
            (src, dst),
            mod_kwargs={'rate': {'e_feat': edge_embedding}}
        )['item'].squeeze(1)
        assert item_latent_factor.shape == (self.item_num, self.embedding_size)
        # prediction
        res_embedding = {'user': user_latent_factor, 'item': item_latent_factor}
        pos_score = self.pred(pos_pred_g, res_embedding, 'rate')
        neg_score = self.pred(neg_pred_g, res_embedding, 'rate')
        assert pos_score.shape[1] == 1
        return pos_score, neg_score