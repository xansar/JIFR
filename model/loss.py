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
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from .TrustSVDmodel import RegLoss
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

class TrustSVDLoss(nn.Module):
    def __init__(self, reduction='mean', lamda=5e-4, lamda_t=5e-4):
        self.reduction = reduction
        super(TrustSVDLoss, self).__init__()
        self.mseloss = MSELoss(reduction=reduction)
        self.regloss = RegLoss(lamda=lamda, lamda_t=lamda_t)

    def forward(self, output, y):
        pred_rate = output['pred_rate'].reshape(-1)
        pred_link = output['pred_link'].reshape(-1)
        rate_mse_loss = self.mseloss(pred_rate, y)
        link_mse_loss = self.mseloss(pred_link, torch.ones_like(pred_link, device=pred_link.device))
        reg_loss = self.regloss(output)
        return rate_mse_loss, link_mse_loss, reg_loss

