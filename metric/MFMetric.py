#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ExtendedEpinionsMF.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/25 11:38   zxx      1.0         None
"""


# import lib
import numpy as np
import torch

class BaseMetric:
    def __init__(self, ks, metric_name):
        self.metric_dict = {}
        self.metric_name = metric_name
        self.ks = ks
        self.init_metrics(metric_name)

    def init_metrics(self, metric_name):
        self.metric_name = metric_name
        self.metric_dict = {}
        for m in self.metric_name:
            self.metric_dict[m] = {}
            for k in self.ks:
                self.metric_dict[m][k] = {'value': 0., 'best': 0., 'cnt': 0}

    def clear_metrics(self):
        for m in self.metric_name:
            for k in self.ks:
                self.metric_dict[m][k]['value'] = 0.
                self.metric_dict[m][k]['cnt'] = 0

    def compute_metrics(self, *input_):
        pass

    def get_batch_metrics(self, *input_):
        pass

    def print_best_metrics(self):
        metric_str = ''
        for m in self.metric_name:
            for k in self.ks:
                v = self.metric_dict[m][k]['best']
                metric_str += f'best: {m}@{k}: {v:.4f}\t'
            metric_str += '\n'
        return metric_str

class MFMetric(BaseMetric):
    def __init__(self, ks, metric_name):
        self.is_early_stop = False
        super(MFMetric, self).__init__(ks, metric_name)

    def _compute_HR(self, total_pred):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # 0位置就是正样本
            if 0 in topk_id:
                self.metric_dict['HR'][k]['value'] += 1
            self.metric_dict['HR'][k]['cnt'] += 1

    def compute_metrics(self, pos_pred, neg_pred):
        total_pred = torch.cat([pos_pred, neg_pred], dim=0)
        self._compute_HR(total_pred)

    def get_batch_metrics(self):
        for k in self.ks:
            self.metric_dict['HR'][k]['value'] /= self.metric_dict['HR'][k]['cnt']
            self.metric_dict['HR'][k]['cnt'] = -1
            if self.metric_dict['HR'][k]['value'] > self.metric_dict['HR'][k]['best']:
                self.metric_dict['HR'][k]['best'] = self.metric_dict['HR'][k]['value']
            elif k == self.ks[-1] and self.metric_dict['HR'][k]['value'] < self.metric_dict['HR'][k]['best']:
                self.is_early_stop = True