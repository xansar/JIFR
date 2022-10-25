#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ExtendedEpinionsMF.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 21:00   zxx      1.0         None
"""

# import lib
import torch.nn as nn
import torch

class MFModel(nn.Module):
    def __init__(self, config):
        super(MFModel, self).__init__()
        self.config = config
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.U = nn.Parameter(torch.randn(self.user_num, self.embedding_size))
        self.I = nn.Parameter(torch.randn(self.item_num, self.embedding_size))

    def forward(self, **inputs):
        user = inputs['user']
        item = inputs['item']
        u = self.U[user]
        i = self.I[item]
        pred = torch.sum(u * i, dim=1)
        return pred

if __name__ == '__main__':
    user = torch.tensor([0, 0, 1, 2])
    item = torch.tensor([0, 1, 1, 0])
    rate = torch.tensor([3, 2, 4, 5], dtype=torch.float)
    model = MFModel(3, 2, 4)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(lr=1e-1, params=model.parameters(), weight_decay=1e-4)
    for i in range(100):
        optimizer.zero_grad()
        pred = model(user=user, item=item)
        loss = loss_func(pred, rate)
        print(f'{i}: {loss.item()}')
        loss.backward()
        optimizer.step()