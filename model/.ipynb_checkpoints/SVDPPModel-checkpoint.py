#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model.py.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/19 14:18   zxx      1.0         None
"""
import torch
import torch.nn as nn
import pandas
class SVDPP(nn.Module):
    def __init__(self, embedding_size=10, user_num=12770, item_num=23143, global_bias=3.95, device='cuda'):
        super(SVDPP, self).__init__()
        # 四组参数：P、Q、W、Y，前两组显式，后两组隐式
        ## 用户显式
        self.P = nn.Embedding(user_num, embedding_size)
        ## 物品显式
        self.Q = nn.Embedding(item_num, embedding_size)
        ## 用户隐式
        self.W = nn.Embedding(user_num + 1, embedding_size, padding_idx=user_num)
        ## 物品隐式
        self.Y = nn.Embedding(item_num + 1, embedding_size, padding_idx=item_num)
        ## 用户偏置
        self.B_u = nn.Embedding(user_num, 1)
        ## 物品偏置
        self.B_i = nn.Embedding(item_num, 1)
        ## 全局偏置
        self.global_bias = nn.Parameter(torch.tensor(global_bias), requires_grad=True)

        self.device = device
    def forward(self, inputs):
        # for k, v in inputs.items():
        #     inputs[k] = v.to(self.device)
        users = inputs['users'].to(self.device)
        items = inputs['items'].to(self.device)
        rated_items = inputs['rated_items'].to(self.device) # 这里idx是+1的，因为0用来mask，所以
        items_nums = inputs['items_nums'].to(self.device)
        I_u_factor = (1 / torch.sqrt(items_nums.unsqueeze(1) + 1e-8)).clamp(max=1)
        p = self.P(users)
        q = self.Q(users)

        y = torch.sum(self.Y(rated_items), dim=1)
        b_u = self.B_u(users)
        b_i = self.B_i(items)
        user_represetation = p + I_u_factor * y 
        pred_rate = torch.sum(q * user_represetation, dim=1, keepdim=True) + b_u + b_i + self.global_bias

        # 用于正则化
        y_i = self.Y(items)
        user_rate_i_num = inputs['user_rate_i_num'].to(self.device)
        U_i_factor = (1 / torch.sqrt(user_rate_i_num.unsqueeze(1) + 1e-8)).clamp(max=1)
        return {
            'pred_rate': pred_rate,
            'b_u': b_u,
            'b_i': b_i,
            'I_u_factor': I_u_factor,
            'U_i_factor': U_i_factor,
            'p_u': p,
            'q_i': q,
            'y_i': y_i,
        }

class RegLoss(nn.Module):
    def __init__(self, lamda=5e-4, lamda_t=5e-4):
        super(RegLoss, self).__init__()
        self.lamda = lamda
        self.lamda_t = lamda_t

    def forward(self, inputs):

        b_u = inputs['b_u']
        I_u_factor = inputs['I_u_factor']
        b_u_reg = self.lamda * torch.sum(torch.square(b_u) * I_u_factor)

        ##
        b_i = inputs['b_i']
        # 评论过这个物品的用户数量
        U_i_factor = inputs['U_i_factor']
        b_i_reg = torch.sum(torch.square(b_i) * U_i_factor)

        ##
        p_u = inputs['p_u']
        p_u_reg = torch.sum(torch.square(torch.norm(p_u)) * I_u_factor)

        ##
        q_i = inputs['q_i']
        q_i_reg = torch.sum(torch.square(q_i) * U_i_factor)

        ##
        y_i = inputs['y_i']
        y_i_reg = torch.sum(torch.square(y_i) * U_i_factor)

        return b_u_reg + b_i_reg + p_u_reg + q_i_reg + y_i_reg

if __name__ == '__main__':
    from dataset import Epinions, collate_fn
    dataset = Epinions()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    device = 'cuda'
    model = SVDPP(embedding_size=10, device=device, global_bias=3.6).to(device)
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






