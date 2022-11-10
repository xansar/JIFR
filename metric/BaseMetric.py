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
    def __init__(self, config):
        self.config = config
        self.is_early_stop = False
        self.early_stop_num = eval(config['OPTIM']['early_stop_num'])
        self.early_stop_cnt = 0
        self.metric_cnt = 0
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])
        self.early_stop_last = -1

        self.is_save = False
        self.metric_dict = {}
        task = config['TRAIN']['task']
        if task == 'Joint':
            self.task_lst = ['Rate', 'Link']
        else:
            self.task_lst = [task]

        metric_name = eval(config['METRIC']['metric_name'])
        self.ks = eval(config['METRIC']['ks'])
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
        # 随机将pos插入一个位置idx
        target_idx = torch.randint(0, neg_pred.shape[1] + 1, (1,))
        total_pred = torch.cat([neg_pred[:, :target_idx], pos_pred, neg_pred[:, target_idx:]], dim=1)
        for m in self.metric_name:
            eval(f'self._compute_{m}')(total_pred, target_idx, task)

    def get_batch_metrics(self):
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    self.metric_dict[t][m][k]['value'] /= self.metric_dict[t][m][k]['cnt']
                    self.metric_dict[t][m][k]['cnt'] = -1
                    if self.metric_dict[t][m][k]['value'] > self.metric_dict[t][m][k]['best']:
                        self.metric_dict[t][m][k]['best'] = self.metric_dict[t][m][k]['value']
                        if k == self.ks[-1] and t == 'Rate' and m == 'HR':
                            self.is_save = True
                    elif k == self.ks[-1] and t == 'Rate' and m == 'HR':
                        self.metric_cnt += 1
                        if self.metric_dict[t][m][k]['value'] < self.early_stop_last and self.metric_cnt * self.eval_step > self.warm_epoch:
                            self.early_stop_cnt += 1
                            if self.early_stop_cnt > self.early_stop_num:
                                self.is_early_stop = True
                        elif self.metric_dict[t][m][k]['value'] >= self.early_stop_last:
                            self.early_stop_cnt = 0
                        self.early_stop_last = self.metric_dict[t][m][k]['value']


    def print_best_metrics(self):
        metric_str = ''
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    v = self.metric_dict[t][m][k]['best']
                    metric_str += f'{t} best: {m}@{k}: {v:.4f}\t'
                metric_str += '\n'
        return metric_str

    def _compute_HR(self, total_pred, target_idx, task):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # target_idx位置就是正样本
            hit = (topk_id == target_idx).count_nonzero(dim=1).clamp(max=1).sum()
            self.metric_dict[task]['HR'][k]['value'] = hit.item()
            self.metric_dict[task]['HR'][k]['cnt'] = total_pred.shape[0]

    def _compute_nDCG(self, total_pred, target_idx, task):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # target_idx位置就是正样本
            value, idx = torch.topk((topk_id == target_idx).int(), 1)
            nDCG = (value / torch.log2(idx + 2)).sum()  # 这里加2是因为idx从0开始计数
            self.metric_dict[task]['nDCG'][k]['value'] = nDCG.item()
            self.metric_dict[task]['nDCG'][k]['cnt'] = total_pred.shape[0]
