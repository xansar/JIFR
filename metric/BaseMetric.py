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
    def __init__(self, ks, task, metric_name):
        self.is_early_stop = False
        self.is_save = False
        self.metric_dict = {}
        if task == 'Joint':
            self.task_lst = ['Rate', 'Link']
        else:
            self.task_lst = [task]

        self.metric_name = metric_name
        self.ks = ks
        self.init_metrics(metric_name)

    def init_metrics(self, metric_name):
        self.metric_name = metric_name
        self.metric_dict = {}
        for t in self.task_lst:
            self.metric_dict[t] = {}
            for m in self.metric_name:
                self.metric_dict[t][m] = {}
                for k in self.ks:
                    self.metric_dict[t][m][k] = {'value': 0., 'best': 0., 'cnt': 0}

    def clear_metrics(self):
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    self.metric_dict[t][m][k]['value'] = 0.
                    self.metric_dict[t][m][k]['cnt'] = 0

    def compute_metrics(self, *input_):
        pass

    def get_batch_metrics(self):
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    self.metric_dict[t][m][k]['value'] /= self.metric_dict[t][m][k]['cnt']
                    self.metric_dict[t][m][k]['cnt'] = -1
                    if self.metric_dict[t][m][k]['value'] > self.metric_dict[t][m][k]['best']:
                        self.metric_dict[t][m][k]['best'] = self.metric_dict[t][m][k]['value']
                        if k == self.ks[-1]:
                            self.is_save = True
                    elif k == self.ks[-1] and self.metric_dict[t][m][k]['value'] < self.metric_dict[t][m][k]['best']:
                        self.is_early_stop = True

    def print_best_metrics(self):
        metric_str = ''
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    v = self.metric_dict[t][m][k]['best']
                    metric_str += f'{t} best: {m}@{k}: {v:.4f}\t'
                metric_str += '\n'
        return metric_str
