#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 21:00   zxx      1.0         None
"""

# import lib
import torch.nn as nn
import torch

class MF(nn.Module):
    def __init__(self, user_num=1597, item_num=24984, embedding_size=10):
        super(MF, self).__init__()
        self.U = nn.Parameter(torch.randn(user_num, embedding_size))
        self.I = nn.Parameter(torch.randn(item_num, embedding_size))

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
    model = MF(3, 2, 4)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(lr=1e-1, params=model.parameters(), weight_decay=1e-4)
    for i in range(100):
        optimizer.zero_grad()
        pred = model(user=user, item=item)
        loss = loss_func(pred, rate)
        print(f'{i}: {loss.item()}')
        loss.backward()
        optimizer.step()