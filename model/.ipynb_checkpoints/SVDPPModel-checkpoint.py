#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   TrustSVDmodel.py.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/19 14:18   zxx      1.0         None
"""
import torch
import torch.nn as nn

class SVDPPModel(nn.Module):
    def __init__(self, config):
        super(SVDPPModel, self).__init__()
        self.config = config
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        global_bias = eval(config['MODEL']['global_bias'])

        if self.task == 'Rate':
            self.nodes_num = self.user_num + self.item_num
        elif self.task == 'Link':
            self.nodes_num = self.total_user_num
        # 四组参数：P、Q、W、Y，前两组显式，后两组隐式
        ## 用户显式
        self.P = nn.Embedding(self.user_num, self.embedding_size)
        ## 物品显式
        self.Q = nn.Embedding(self.item_num, self.embedding_size)
        ## 用户隐式
        self.W = nn.Embedding(self.total_user_num + 1, self.embedding_size, padding_idx=self.total_user_num)
        ## 物品隐式
        self.Y = nn.Embedding(self.item_num + 1, self.embedding_size, padding_idx=self.item_num)
        ## 用户偏置
        self.B_u = nn.Embedding(self.user_num, 1)
        ## 物品偏置
        self.B_i = nn.Embedding(self.item_num, 1)
        ## 全局偏置
        self.global_bias = nn.Parameter(torch.tensor(global_bias), requires_grad=True)

        self.device = config['TRAIN']['device']

    def forward(self, inputs, neg_num=None):
        users = inputs['users'].to(self.device, non_blocking=True)
        items = inputs['items'].to(self.device, non_blocking=True)
        rated_items = inputs['rated_items'].to(self.device, non_blocking=True)
        items_nums = inputs['items_nums'].to(self.device, non_blocking=True)
        I_u_factor = (1 / torch.sqrt(items_nums.unsqueeze(1) + 1e-8)).clamp(max=1)
        trusts = inputs['trusts'].to(self.device, non_blocking=True)
        trusts_nums = inputs['trusts_nums'].to(self.device, non_blocking=True)
        T_u_factor = (1 / torch.sqrt(trusts_nums.unsqueeze(1) + 1e-8)).clamp(max=1)
        p = self.P(users)
        q = self.Q(items)

        w = torch.sum(self.W(trusts), dim=1)
        y = torch.sum(self.Y(rated_items), dim=1)

        b_u = self.B_u(users)
        b_i = self.B_i(items)
        user_represetation = p + I_u_factor * y + T_u_factor * w
        # 负采样预测前要处理一下，把user向量重复得跟item向量一样长
        if neg_num is not None:
            user_represetation = user_represetation.repeat(neg_num, 1)
            b_u = b_u.repeat(neg_num, 1)
        pred_rate = torch.sum(q * user_represetation, dim=1, keepdim=True) + b_u + b_i + self.global_bias
        res_dicet = {
            'pred_rate': pred_rate,
        }
        if neg_num is None: # 正样本计算时返回正则化项需要的变量
            # trust矩阵分解
            w_v = self.W(trusts[:, 0])
            pred_link = torch.sum(w_v * p, dim=1)
            # 正则化项
            w_u = self.W(users)
            y_i = self.Y(items)
            user_rate_i_num = inputs['user_rate_i_num'].to(self.device, non_blocking=True)
            U_i_factor = (1 / torch.sqrt(user_rate_i_num.unsqueeze(1) + 1e-8)).clamp(max=1)
            user_trust_u_num = inputs['user_trust_u_num'].to(self.device, non_blocking=True)
            T_u_plus_factor = (1 / torch.sqrt(user_trust_u_num.unsqueeze(1) + 1e-8)).clamp(max=1)
            res_dicet.update({
                'pred_link': pred_link,
                'b_u': b_u,
                'b_i': b_i,
                'I_u_factor': I_u_factor,
                'T_u_factor': T_u_factor,
                'p_u': p,
                'q_i': q,
                'U_i_factor': U_i_factor,
                'T_u_plus_factor': T_u_plus_factor,
                'y_i': y_i,
                'w_u': w_u
            })
        return res_dicet

class RegLoss(nn.Module):
    def __init__(self, lamda=5e-4, lamda_t=5e-4):
        super(RegLoss, self).__init__()
        self.lamda = lamda
        self.lamda_t = lamda_t

    def forward(self, inputs):
        b_u = inputs['b_u']
        I_u_factor = inputs['I_u_factor']
        b_u_reg = self.lamda * torch.mean(torch.square(b_u) * I_u_factor)

        ##
        b_i = inputs['b_i']
        # 评论过这个物品的用户数量
        U_i_factor = inputs['U_i_factor']
        b_i_reg = self.lamda * torch.mean(torch.square(b_i) * U_i_factor)

        ##
        p_u = inputs['p_u']
        T_u_factor = inputs['T_u_factor']
        p_u_reg = torch.mean(torch.square(torch.norm(p_u)) * (self.lamda * I_u_factor + self.lamda_t * T_u_factor))

        ##
        q_i = inputs['q_i']
        q_i_reg = self.lamda * torch.mean(torch.square(q_i) * U_i_factor)

        ##
        y_i = inputs['y_i']
        y_i_reg = self.lamda * torch.mean(torch.square(y_i) * U_i_factor)

        return b_u_reg + b_i_reg + p_u_reg + q_i_reg + y_i_reg 

if __name__ == '__main__':
    from dataset import Epinions, collate_fn
    dataset = Epinions()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    device = 'cuda'
    model = TrustSVD(embedding_size=10, device=device, global_bias=3.6).to(device)
    loss_func = nn.MSELoss().to(device)
    reg_func = RegLoss().to(device)
    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())
    from tqdm import tqdm

    for e in range(3):
        all_loss = torch.tensor(0.0)
        for idx, x in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            y = x['scores'].to(device)
            output = model(x)
            loss = loss_func(output["pred_rate"].reshape(-1), y)
            all_loss += loss.item()
            loss += reg_func(output)
            loss.backward()
            optimizer.step()
        print(f'{e}: {torch.sqrt(all_loss).item() / idx:.4f}')






