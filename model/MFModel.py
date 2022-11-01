#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ExtendedEpinionsMF.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 21:00   zxx      1.0         None
"""

import torch
# import lib
import torch.nn as nn

class MFModel(nn.Module):
    def __init__(self, config):
        super(MFModel, self).__init__()
        self.config = config
        self.task = config['TRAIN']['task']
        self.user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.embedding_size = eval(config['MODEL']['embedding_size'])

        if self.task == 'Rate':
            self.nodes_num = self.user_num + self.item_num
        elif self.task == 'Link':
            self.nodes_num = self.total_user_num
        self.embedding = nn.Embedding(self.nodes_num, self.embedding_size)
    #     self.weight_init()
    #
    # def weight_init(self):
    #     nn.init.xavier_normal_(self.embedding.data)

    def forward(self, **inputs):
        u = inputs['u']
        # 因为user和item的embedding放在一起，所以item部分的id需要加上user_num
        if self.task == 'Rate':
            v = inputs['v'] + self.user_num
        else:
            v = inputs['v']
        u = self.embedding(u)
        v = self.embedding(v)
        pred = torch.sum(u * v, dim=1)
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