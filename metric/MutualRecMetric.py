#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MutualRecMetric.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/30 16:22   zxx      1.0         None
"""

# import lib
import torch
from .BaseMetric import BaseMetric

class MutualRecMetric(BaseMetric):
    def __init__(self, ks, task, metric_name):
        super(MutualRecMetric, self).__init__(ks, task, metric_name)

    def _compute_HR(self, total_pred, task):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # 0位置就是正样本
            hit = (topk_id == 0).count_nonzero(dim=1).clamp(max=1).sum()
            self.metric_dict[task]['HR'][k]['value'] = hit.item()
            self.metric_dict[task]['HR'][k]['cnt'] = total_pred.shape[0]

    def compute_metrics(self, pos_pred, neg_pred, task='Rate'):
        # total_pred, bsz * 101, 第一维是正样本预测值
        total_pred = torch.cat([pos_pred, neg_pred], dim=1)
        self._compute_HR(total_pred, task)


