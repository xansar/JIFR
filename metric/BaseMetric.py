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
    def __init__(self, config, bin_sep_lst):
        self.config = config
        self.bin_sep_lst = bin_sep_lst
        # self.is_early_stop = False
        self.is_save = False
        save_metric = config['METRIC'].get('save_metric', 'nDCG@10')
        self.save_metric, self.indicator_k = save_metric.split('@')
        self.indicator_k = eval(self.indicator_k)

        # self.early_stop_num = eval(config['OPTIM']['early_stop_num'])
        # self.early_stop_cnt = 0
        self.metric_cnt = 0
        self.iter_step = 0
        self.eval_step = eval(config['TRAIN']['eval_step'])
        if not 'step_per_epoch' in config['TRAIN'].keys():
            assert 'eval_batch_size' in config['DATA'].keys()
            self.batch_size = eval(config['DATA']['eval_batch_size'])
        else:
            self.batch_size = None
        # self.warm_epoch = eval(config['TRAIN']['warm_epoch'])
        # self.early_stop_last = -1

        self.metric_dict = {}
        task = config['TRAIN']['task']
        if task == 'Joint':
            self.task_lst = ['Rate', 'Link']
            self.indicator_task = 'Rate'
        else:
            self.task_lst = [task]
            self.indicator_task = task

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
                        for key in ['bin_value', 'bin_best', 'bin_cnt']:
                            # 三个key分别记录当前值，最好值，当前计数
                            # 每个key下面是一个n*n的表格，每个格子表示一个分组
                            self.metric_dict[t][m][k].update(
                                {key:
                                     [
                                        [0. for _ in range(len(self.bin_sep_lst) + 1)]  # 这里+1用来计算行列汇总
                                        for _ in range(len(self.bin_sep_lst) + 1)
                                     ]
                                 }
                            )

                        # self.metric_dict[t][m][k].update({'bin_value': [0. for _ in range(len(self.bin_sep_lst))]})
                        # self.metric_dict[t][m][k].update({'bin_best': [0. for _ in range(len(self.bin_sep_lst))]})
                        # self.metric_dict[t][m][k].update({'bin_cnt': [0. for _ in range(len(self.bin_sep_lst))]})


    def clear_metrics(self):
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    self.metric_dict[t][m][k]['value'] = 0.
                    self.metric_dict[t][m][k]['cnt'] = 0

                    if self.bin_sep_lst is not None:
                        self.metric_dict[t][m][k]['bin_value'] = [
                                        [0. for _ in range(len(self.bin_sep_lst) + 1)]
                                        for _ in range(len(self.bin_sep_lst) + 1)
                                     ]
                        self.metric_dict[t][m][k]['bin_cnt'] = [
                                        [0. for _ in range(len(self.bin_sep_lst) + 1)]
                                        for _ in range(len(self.bin_sep_lst) + 1)
                                     ]

                        # self.metric_dict[t][m][k]['bin_cnt'] = [0. for _ in range(len(self.bin_sep_lst))]
                        # self.metric_dict[t][m][k]['bin_value'] = [0. for _ in range(len(self.bin_sep_lst))]

    def compute_metrics(self, batch_users, rating_k, gt, task, is_test=False):
        mask = torch.zeros_like(rating_k, dtype=torch.float)
        iDCG_mask = torch.zeros_like(rating_k, dtype=torch.float)
        for i in range(len(rating_k)):
            single_gt = torch.tensor(gt[i], dtype=torch.long)
            if len(single_gt) >= rating_k.shape[1]:
                gt_len = rating_k.shape[1]
            else:
                gt_len = len(single_gt)
            iDCG_mask[i, :gt_len] = 1.

            rating = rating_k[i]
            # 计算rating的每一位是否是gt，如果是的话，为1，否则为0
            mask[i] = torch.isin(rating, single_gt).float()

        self._compute_Recall(batch_users, mask, task, is_test, gt=gt)
        self._compute_nDCG(batch_users, mask, task, is_test, iDCG_mask=iDCG_mask)
        self._compute_Precision(batch_users, mask, task, is_test)
        # for m in self.metric_name:
        #     eval(f'self._compute_{m}')(batch_users, mask, task)

    def get_batch_metrics(self, is_test=False):
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
                        if k == self.indicator_k and t == task and m == self.save_metric:
                            self.is_save = True
                    # elif k == self.indicator_k and t == task and m == self.save_metric:
                    #     self.metric_cnt += 1
                    #     if self.metric_dict[t][m][k]['value'] < self.early_stop_last and self.metric_cnt * self.eval_step > self.warm_epoch:
                    #         self.early_stop_cnt += 1
                    #         if self.early_stop_cnt > self.early_stop_num:
                    #             self.is_early_stop = True
                    #     elif self.metric_dict[t][m][k]['value'] >= self.early_stop_last:
                    #         self.early_stop_cnt = 0
                    #     self.early_stop_last = self.metric_dict[t][m][k]['value']

                    if self.bin_sep_lst is not None and is_test:
                        for i in range(len(self.bin_sep_lst) + 1):
                            for j in range(len(self.bin_sep_lst) + 1):
                                if self.metric_dict[t][m][k]['bin_cnt'][i][j] != 0:
                                    self.metric_dict[t][m][k]['bin_value'][i][j] /= self.metric_dict[t][m][k]['bin_cnt'][i][j]
                                    if self.metric_dict[t][m][k]['bin_value'][i][j] > self.metric_dict[t][m][k]['bin_best'][i][j]:
                                        self.metric_dict[t][m][k]['bin_best'][i][j] = self.metric_dict[t][m][k]['bin_value'][i][j]
                                else:
                                    self.metric_dict[t][m][k]['bin_value'][i][j] = 0.
                        # for i in range(len(self.bin_sep_lst)):
                        #     self.metric_dict[t][m][k]['bin_value'][i] /= self.metric_dict[t][m][k]['bin_cnt'][i]
                        #     if self.metric_dict[t][m][k]['bin_value'][i] > self.metric_dict[t][m][k]['bin_best'][i]:
                        #         self.metric_dict[t][m][k]['bin_best'][i] = self.metric_dict[t][m][k]['bin_value'][i]

    def print_best_metrics(self):
        metric_str = ''
        for t in self.task_lst:
            for m in self.metric_name:
                for k in self.ks:
                    v = self.metric_dict[t][m][k]['best']
                    metric_str += f'{t} best: {m}@{k}: {v:.4f}\t'
                    # if self.bin_sep_lst is not None:
                    #     metric_str += '\n'
                    #     for i in range(len(self.bin_sep_lst)):
                    #         v = self.metric_dict[t][m][k]['bin_best'][i]
                    #         s = self.bin_sep_lst[i]
                    #         if i == len(self.bin_sep_lst) - 1:
                    #             e = '∞'
                    #         else:
                    #             e = self.bin_sep_lst[i + 1]
                    #         metric_str += f'{t} best: {m}@{k} in {s}-{e}: {v:.4f}\t'
                    #     metric_str += '\n'
                metric_str += '\n'
        return metric_str

    def _get_batch_id_lst(self, id_lst):
        if self.batch_size is not None:
            left_eid = self.iter_step * self.batch_size
            right_eid = (self.iter_step + 1) * self.batch_size
            id_lst = torch.masked_select(id_lst, (id_lst >= left_eid) & (id_lst < right_eid))
            # 将原始的user id映射到当前batch
            id_lst = id_lst - left_eid
        return id_lst

    def _compute_Precision(self, batch_users, mask, task, is_test, **kwargs):
        for k in self.ks:
            mask_k = mask[:, :k]
            precision = mask_k.sum(1) / k

            if self.bin_sep_lst is not None and is_test:
                self.compute_bins_metrics(precision, batch_users, task, 'Precision', k)

                # for i in range(len(self.bins_id_lst)):
                #     id_lst = self.bins_id_lst[i]    # 包含了这一组内的用户id
                #     # batch 的情况
                #     # 因此需要把id_lst中这一部分的userid取出来，左闭右开
                #     left_uid = torch.min(batch_users)
                #     right_uid = torch.max(batch_users)
                #     id_lst = torch.masked_select(id_lst, (id_lst >= left_uid) & (id_lst < right_uid))
                #     id_lst = id_lst - left_uid
                #
                #     precision_bin = precision[id_lst].sum()
                #     self.metric_dict[task]['Precision'][k]['bin_value'][i] += precision_bin.item()
                #     self.metric_dict[task]['Precision'][k]['bin_cnt'][i] += id_lst.shape[0]

            self.metric_dict[task]['Precision'][k]['value'] += precision.sum().item()
            self.metric_dict[task]['Precision'][k]['cnt'] += len(mask_k)

    def _compute_Recall(self, batch_users, mask, task, is_test, **kwargs):
        gt = kwargs.get('gt', None)
        for k in self.ks:
            mask_k = mask[:, :k]
            gt_len = torch.tensor([len(g) for g in gt])
            recall = mask_k.sum(1) / gt_len

            if self.bin_sep_lst is not None and is_test:
                self.compute_bins_metrics(recall, batch_users, task, 'Recall', k)

                # for i in range(len(self.bins_id_lst)):
                #     id_lst = self.bins_id_lst[i]    # 包含了这一组内的用户id
                #     # batch 的情况
                #     # 因此需要把id_lst中这一部分的userid取出来，左闭右开
                #     left_uid = torch.min(batch_users)
                #     right_uid = torch.max(batch_users)
                #     id_lst = torch.masked_select(id_lst, (id_lst >= left_uid) & (id_lst < right_uid))
                #     id_lst = id_lst - left_uid
                #
                #     recall_bin = recall[id_lst].sum()
                #     self.metric_dict[task]['Recall'][k]['bin_value'][i] += recall_bin.item()
                #     self.metric_dict[task]['Recall'][k]['bin_cnt'][i] += id_lst.shape[0]

            self.metric_dict[task]['Recall'][k]['value'] += recall.sum().item()
            self.metric_dict[task]['Recall'][k]['cnt'] += len(mask_k)

    def _compute_nDCG(self, batch_users, mask, task, is_test, **kwargs):
        iDCG_mask = kwargs.get('iDCG_mask', None)
        for k in self.ks:
            mask_k = mask[:, :k]
            iDCG_mask_k = iDCG_mask[:, :k]
            DCG = torch.sum(mask_k * (1. / torch.log2(torch.arange(2, k + 2))), dim=1)
            iDCG = torch.sum(iDCG_mask_k * (1. / torch.log2(torch.arange(2, k + 2))), dim=1)
            nDCG = DCG / iDCG
            # value, idx = torch.topk(mask_k.int(), 1)
            # nDCG = (value / torch.log2(idx + 2)) # 这里加2是因为idx从0开始计数

            if self.bin_sep_lst is not None and is_test:
                self.compute_bins_metrics(nDCG, batch_users, task, 'nDCG', k)
                # for i in range(len(self.bins_id_lst)):
                #     id_lst = self.bins_id_lst[i]    # 包含了这一组内的用户id
                #     # batch 的情况
                #     # 因此需要把id_lst中这一部分的userid取出来，左闭右开
                #     left_uid = torch.min(batch_users)
                #     right_uid = torch.max(batch_users)
                #     id_lst = torch.masked_select(id_lst, (id_lst >= left_uid) & (id_lst < right_uid))
                #     id_lst = id_lst - left_uid
                #
                #     nDCG_bin = nDCG[id_lst].sum()
                #     self.metric_dict[task]['nDCG'][k]['bin_value'][i] += nDCG_bin.item()
                #     self.metric_dict[task]['nDCG'][k]['bin_cnt'][i] += id_lst.shape[0]

            self.metric_dict[task]['nDCG'][k]['value'] += nDCG.sum().item()
            self.metric_dict[task]['nDCG'][k]['cnt'] += mask_k.shape[0]

    def compute_bins_metrics(self,
                             metric_value: torch.Tensor,
                             batch_users: torch.Tensor,
                             task: str,
                             metric_name: str,
                             k: int,
                             ):
        for i in range(len(self.bin_sep_lst) + 1): # 分组和汇总（+1）
            for j in range(len(self.bin_sep_lst) + 1):
                id_lst = self.bins_id_lst[i][j]
                # 需要把id_lst中这一部分的userid取出来，左闭右开，需要batch_users有序
                left_uid = torch.min(batch_users)
                right_uid = torch.max(batch_users)
                id_lst = torch.masked_select(id_lst, (id_lst >= left_uid) & (id_lst < right_uid))
                id_lst = id_lst - left_uid

                metric_value_bin = metric_value[id_lst].sum()
                self.metric_dict[task][metric_name][k]['bin_value'][i][j] += metric_value_bin.item()
                self.metric_dict[task][metric_name][k]['bin_cnt'][i][j] += id_lst.shape[0]