#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LightGCNMetric.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/26 16:28   zxx      1.0         None
"""


# import lib
import torch
from .BaseMetric import BaseMetric

class LightGCNMetric(BaseMetric):
    def __init__(self, ks, metric_name):
        super(LightGCNMetric, self).__init__(ks, metric_name)

    def _compute_HR(self, total_pred):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # 0位置就是正样本
            hit = (topk_id == 0).count_nonzero(dim=1).clamp(max=1).sum()
            self.metric_dict['HR'][k]['value'] = hit.item()
            self.metric_dict['HR'][k]['cnt'] = total_pred.shape[0]

    def compute_metrics(self, pos_pred, neg_pred):
        # total_pred, bsz * 101, 第一维是正样本预测值
        total_pred = torch.cat([pos_pred, neg_pred], dim=1)
        self._compute_HR(total_pred)

    def get_batch_metrics(self):
        for k in self.ks:
            self.metric_dict['HR'][k]['value'] /= self.metric_dict['HR'][k]['cnt']
            self.metric_dict['HR'][k]['cnt'] = -1
            if self.metric_dict['HR'][k]['value'] > self.metric_dict['HR'][k]['best']:
                self.metric_dict['HR'][k]['best'] = self.metric_dict['HR'][k]['value']
                if k == self.ks[-1]:
                    self.is_save = True
            elif k == self.ks[-1] and self.metric_dict['HR'][k]['value'] < self.metric_dict['HR'][k]['best']:
                self.is_early_stop = True
