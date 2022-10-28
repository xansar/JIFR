#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   BaseMetric.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/28 18:00   zxx      1.0         None
"""

# import lib
import torch


class BaseMetric:
    def __init__(self, ks, metric_name):
        self.is_early_stop = False
        self.is_save = False
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
