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
        self.bin_sep_lst = eval(config['METRIC'].get('bin_sep_lst', 'None'))
        self.is_early_stop = False
        self.is_save = False
        self.save_metric = config['METRIC'].get('save_metric', 'nDCG')

        self.early_stop_num = eval(config['OPTIM']['early_stop_num'])
        self.early_stop_cnt = 0
        self.metric_cnt = 0
        self.iter_step = 0
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.batch_size = eval(config['DATA']['eval_batch_size'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])
        self.early_stop_last = -1

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

                    if self.bin_sep_lst is not None:
                        self.metric_dict[t][m][k].update({'bin_value': [0. for _ in range(len(self.bin_sep_lst))]})
                        self.metric_dict[t][m][k].update({'bin_best': [0. for _ in range(len(self.bin_sep_lst))]})
                        self.metric_dict[t][m][k].update({'bin_cnt': [0. for _ in range(len(self.bin_sep_lst))]})


    def clear_metrics(self):
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    self.metric_dict[t][m][k]['value'] = 0.
                    self.metric_dict[t][m][k]['cnt'] = 0

                    if self.bin_sep_lst is not None:
                        self.metric_dict[t][m][k]['bin_value'] = [0. for _ in range(len(self.bin_sep_lst))]
                        self.metric_dict[t][m][k]['bin_cnt'] = [0. for _ in range(len(self.bin_sep_lst))]

    def compute_metrics(self, *input_, task):
        pos_pred, neg_pred = input_
        # total_pred, bsz * 101, 第一维是正样本预测值
        # 随机将pos插入一个位置idx
        target_idx = torch.randint(max(self.ks) + 1, neg_pred.shape[1] + 1, (1,))
        total_pred = torch.cat([neg_pred[:, :target_idx], pos_pred, neg_pred[:, target_idx:]], dim=1)
        for m in self.metric_name:
            eval(f'self._compute_{m}')(total_pred, target_idx, task)

    def get_batch_metrics(self):
        if len(self.task_lst) == 2:
            task = 'Rate'
        else:
            task = self.task_lst[0]
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    self.metric_dict[t][m][k]['value'] /= self.metric_dict[t][m][k]['cnt']
                    self.metric_dict[t][m][k]['cnt'] = -1
                    if self.metric_dict[t][m][k]['value'] > self.metric_dict[t][m][k]['best']:
                        self.metric_dict[t][m][k]['best'] = self.metric_dict[t][m][k]['value']
                        if k == self.ks[-1] and t == task and m == self.save_metric:
                            self.is_save = True
                    elif k == self.ks[-1] and t == task and m == self.save_metric:
                        self.metric_cnt += 1
                        if self.metric_dict[t][m][k]['value'] < self.early_stop_last and self.metric_cnt * self.eval_step > self.warm_epoch:
                            self.early_stop_cnt += 1
                            if self.early_stop_cnt > self.early_stop_num:
                                self.is_early_stop = True
                        elif self.metric_dict[t][m][k]['value'] >= self.early_stop_last:
                            self.early_stop_cnt = 0
                        self.early_stop_last = self.metric_dict[t][m][k]['value']

                    if self.bin_sep_lst is not None:
                        for i in range(len(self.bin_sep_lst)):
                            self.metric_dict[t][m][k]['bin_value'][i] /= self.metric_dict[t][m][k]['bin_cnt'][i]
                            if self.metric_dict[t][m][k]['bin_value'][i] > self.metric_dict[t][m][k]['bin_best'][i]:
                                self.metric_dict[t][m][k]['bin_best'][i] = self.metric_dict[t][m][k]['bin_value'][i]

    def print_best_metrics(self):
        metric_str = ''
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    v = self.metric_dict[t][m][k]['best']
                    metric_str += f'{t} best: {m}@{k}: {v:.4f}\t'
                    if self.bin_sep_lst is not None:
                        metric_str += '\n'
                        for i in range(len(self.bin_sep_lst)):
                            v = self.metric_dict[t][m][k]['bin_best'][i]
                            s = self.bin_sep_lst[i]
                            if i == len(self.bin_sep_lst) - 1:
                                e = '∞'
                            else:
                                e = self.bin_sep_lst[i + 1]
                            metric_str += f'{t} best: {m}@{k} in {s}-{e}: {v:.4f}\t'
                        metric_str += '\n'
                metric_str += '\n'
        return metric_str

    def _compute_HR(self, total_pred, target_idx, task):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # target_idx位置就是正样本
            ## total hit
            hit = (topk_id == target_idx).count_nonzero(dim=1).clamp(max=1)
            # 下面是看分组的metric，思路就是将该组对应的边取出来计算metric
            if self.bin_sep_lst is not None:
                ## bin hit
                for i in range(len(self.bins_id_lst)):
                    id_lst = self.bins_id_lst[i]    # 每一个id_lst都是一个tensor
                    # 使用全图的情况：hit_bin = hit[id_lst].sum()

                    # batch 的情况
                    # 这一批次的eid范围是 self.iter_step * batch_size: (self.iter_step + 1) * batch_size
                    # 因此需要把id_lst中这一部分的eid取出来，左闭右开
                    left_eid = self.iter_step * self.batch_size
                    right_eid = (self.iter_step + 1) * self.batch_size
                    id_lst = torch.masked_select(id_lst, (id_lst >= left_eid) & (id_lst < right_eid))
                    # 将原始的边id映射到当前batch
                    id_lst = id_lst - left_eid

                    hit_bin = hit[id_lst].sum()
                    self.metric_dict[task]['HR'][k]['bin_value'][i] += hit_bin.item()
                    self.metric_dict[task]['HR'][k]['bin_cnt'][i] += id_lst.shape[0]

            self.metric_dict[task]['HR'][k]['value'] += hit.sum().item()
            self.metric_dict[task]['HR'][k]['cnt'] += total_pred.shape[0]

    def _compute_nDCG(self, total_pred, target_idx, task):
        for k in self.ks:
            _, topk_id = torch.topk(total_pred, k)
            # target_idx位置就是正样本
            value, idx = torch.topk((topk_id == target_idx).int(), 1)
            nDCG = (value / torch.log2(idx + 2))  # 这里加2是因为idx从0开始计数

            ## bin nDCG
            if self.bin_sep_lst is not None:
                for i in range(len(self.bins_id_lst)):
                    id_lst = self.bins_id_lst[i]
                    # batch 的情况
                    # 这一批次的eid范围是 self.iter_step * batch_size: (self.iter_step + 1) * batch_size
                    # 因此需要把id_lst中这一部分的eid取出来，左闭右开
                    left_eid = self.iter_step * self.batch_size
                    right_eid = (self.iter_step + 1) * self.batch_size
                    id_lst = torch.masked_select(id_lst, (id_lst >= left_eid) & (id_lst < right_eid))
                    # 将原始的边id映射到当前batch
                    id_lst = id_lst - left_eid

                    nDCG_bin = nDCG[id_lst].sum()
                    self.metric_dict[task]['nDCG'][k]['bin_value'][i] += nDCG_bin.item()
                    self.metric_dict[task]['nDCG'][k]['bin_cnt'][i] += id_lst.shape[0]

            self.metric_dict[task]['nDCG'][k]['value'] += nDCG.sum().item()
            self.metric_dict[task]['nDCG'][k]['cnt'] += total_pred.shape[0]
