#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   loss.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/26 16:55   zxx      1.0         None
"""

# import lib
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)

class BPRLoss(nn.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        super(BPRLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        if self.reduction == 'mean':
            reduce_func = torch.mean
        elif self.reduction == 'sum':
            reduce_func = torch.sum
        loss = reduce_func((-torch.log(torch.sigmoid(pos_score - neg_score).clamp(min=1e-8))).clamp(max=20))
        return loss

