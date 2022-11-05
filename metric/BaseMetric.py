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

    def compute_metrics(self, *input_, task):
        pos_pred, neg_pred = input_
        # total_pred, bsz * 101, 第一维是正样本预测值
        total_pred = torch.cat([pos_pred, neg_pred], dim=1)
        for m in self.metric_name:
            eval(f'self._compute_{m}')(total_pred, task)

    def get_batch_metrics(self):
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    self.metric_dict[t][m][k]['value'] /= self.metric_dict[t][m][k]['cnt']
                    self.metric_dict[t][m][k]['cnt'] = -1
                    if self.metric_dict[t][m][k]['value'] > self.metric_dict[t][m][k]['best']:
                        self.metric_dict[t][m][k]['best'] = self.metric_dict[t][m][k]['value']
                        if k == self.ks[-1] and t == 'Rate':
                            self.is_save = True
                    elif k == self.ks[-1] \
                            and self.metric_dict[t][m][k]['value'] < self.metric_dict[t][m][k]['best'] \
                            and t == 'Rate':
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

    def _compute_HR(self, total_pred, task):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # 0位置就是正样本
            hit = (topk_id == 0).count_nonzero(dim=1).clamp(max=1).sum()
            self.metric_dict[task]['HR'][k]['value'] = hit.item()
            self.metric_dict[task]['HR'][k]['cnt'] = total_pred.shape[0]

    def _compute_nDCG(self, total_pred, task):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # 0位置就是正样本
            value, idx = torch.topk((topk_id == 0).int(), 1)
            nDCG = (value / torch.log2(idx + 2)).sum()  # 这里加2是因为idx从0开始计数
            self.metric_dict[task]['nDCG'][k]['value'] = nDCG.item()
            self.metric_dict[task]['nDCG'][k]['cnt'] = total_pred.shape[0]
