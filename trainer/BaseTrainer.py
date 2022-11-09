#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   BaseTrainer.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/28 17:15   zxx      1.0         None
"""


import torch
import numpy as np
import numba
from numba.typed import List
from tqdm import tqdm
import os
import json
import dgl

"""
BaseTrainer主要用来写一些通用的函数，比如打印config之类
"""

@numba.njit()
def neg_sampling(u_lst, pos_lst, neg_num, total_num):
    lst = List()
    lst.append(0)
    lst.pop()
    for u in u_lst:
        history = pos_lst[u]
        for _ in range(neg_num):
            j = np.random.randint(total_num)
            while j in history:
                j = np.random.randint(total_num)
            lst.append(j)
    return lst

class BaseTrainer:
    def __init__(self, config):
        self.task = config['TRAIN']['task']

        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        # 设置log地址
        self.model_name = self.config['MODEL']['model_name']
        self.data_name = config['DATA']['data_name']
        log_dir = self.config['TRAIN']['log_pth']
        if not os.path.isdir(os.path.join(log_dir, self.model_name, self.data_name)):
            os.makedirs(os.path.join(log_dir, self.model_name, self.data_name))
        self.log_pth = os.path.join(log_dir, self.model_name, self.data_name, f'{self.task}_{self.random_seed}_{self.model_name}.txt')
        # 设置保存地址
        save_dir = self.config['TRAIN']['save_pth']
        if not os.path.isdir(os.path.join(save_dir, self.model_name, self.data_name)):
            os.makedirs(os.path.join(save_dir, self.model_name, self.data_name))
        self.save_pth = os.path.join(save_dir, self.model_name, self.data_name, f'{self.task}_{self.random_seed}_{self.model_name}.pth')
        # 打印config
        self._print_config()

        self.get_history_lst()

    def get_history_lst(self):
        with open(f'./data/{self.data_name}/splited_data/user2v.json', 'r', encoding='utf-8') as f:
            user2v = json.load(f)

        user2item = user2v['user2item']
        user2trust = user2v['user2trust']
        u_lst = []
        item_lst = []
        trust_lst = []
        assert user2trust.keys() == user2item.keys()

        if self.task == 'Rate' or self.task == 'Joint':
            max_len = -1
            for u, v in user2item.items():
                u_lst.append(int(u))
                item_lst.append(v)
                if len(v) > max_len:
                    max_len = len(v)
            for i in range(len(item_lst)):
                item_lst[i] = item_lst[i] + [-1 for _ in range(max_len - len(item_lst[i]))]
            self.u_lst = np.array(u_lst, dtype=np.int32)
            self.item_lst = np.array(item_lst, dtype=np.int32)

        if self.task == 'Link' or self.task == 'Joint':
            max_len = -1
            for u, v in user2trust.items():
                if self.task == 'Link':
                    u_lst.append(int(u))
                trust_lst.append(v)
                if len(v) > max_len:
                    max_len = len(v)
            for i in range(len(trust_lst)):
                trust_lst[i] = trust_lst[i] + [-1 for _ in range(max_len - len(trust_lst[i]))]
            if self.task == 'Link':
                self.u_lst = np.array(u_lst, dtype=np.int32)
            self.trust_lst = np.array(item_lst, dtype=np.int32)


    def _print_config(self):
        # 用来打印config信息
        config_str = ''
        config_str += '=' * 10 + "Config" + '=' * 10 + '\n'
        for k, v in self.config.items():
            config_str += k + ': \n'
            for _k, _v in v.items():
                config_str += f'\t{_k}: {_v}\n'
        config_str += ('=' * 25 + '\n')
        tqdm.write(self._log(config_str, mode='w'))

    def _to(self, device=None):
        # 整体迁移
        if device is None:
            self.model = self.model.to(self.config['TRAIN']['device'])
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
            self.config['TRAIN']['device'] = device

    def _generate_metric_str(self, metric_str):
        # 根据metric结果，生成文本
        for t in self.metric.metric_dict.keys():
            for m in self.metric.metric_dict[t].keys():
                for k in self.metric.metric_dict[t][m].keys():
                    v = self.metric.metric_dict[t][m][k]['value']
                    metric_str += f'{t} {m}@{k}: {v:.4f}\t'
                metric_str += '\n'
        self.metric.clear_metrics()
        return metric_str

    def _log(self, str_, mode='a'):
        # 将log写入文件
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def _save_model(self, save_pth):
        # 保存最好模型
        tqdm.write("Save Best Model!")
        dir_pth, _ = os.path.split(save_pth)
        if not os.path.isdir(dir_pth):
            father_dir_pth, _ = os.path.split(dir_pth)
            if not os.path.isdir(father_dir_pth):
                os.mkdir(father_dir_pth)
            os.mkdir(dir_pth)
        torch.save(self.model.state_dict(), save_pth)

    def _load_model(self, save_pth, strict=False):
        tqdm.write("Load Best Model!")
        state_dict = torch.load(save_pth)
        self.model.load_state_dict(state_dict, strict=strict)

    def construct_negative_graph(self, graph, k, etype):
        utype, _, vtype = etype
        src, dst = graph.edges(etype=etype)
        neg_src = src.repeat_interleave(k)
        if vtype == utype:
            neg_dst = neg_sampling(self.u_lst, self.trust_lst, k, self.total_user_num)
        else:
            neg_dst = neg_sampling(self.u_lst, self.item_lst, k, self.item_num)

        neg_dst = torch.tensor(neg_dst, device=neg_src.device)
        neg_dst = neg_dst.reshape(-1, k)[src].reshape(-1)

        assert neg_dst.shape[0] == len(src) * k
        # neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,), device=neg_src.device)
        return dgl.heterograph(
            {etype: (neg_src, neg_dst)},
            num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

    """
    下面的函数需要被overwrite
    """
    def step(self, mode='train', **inputs):
        # should be overwrited
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            """
            write codes for model forward computation
            """
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            return loss.item()
        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                """
                write codes for model forward computation
                """
                loss = self.loss_func(pos_pred, weight)
                self.metric.compute_metrics(...)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def train(self):
        # 整体训练流程
        tqdm.write(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])

        for e in range(1, epoch + 1):
            all_loss = 0.0
            """
            write codes for train a epoch
            """
            metric_str = f'Train Epoch: {e}\nLoss: {all_loss:.4f}\n'

            if e % self.eval_step == 0:
                self.metric.clear_metrics()
                all_loss = 0.0
                """
                write codes for evaluate a epoch
                """
                self.metric.get_batch_metrics()
                metric_str += f'Evaluate Epoch: {e}\n'
                metric_str += f'loss: {all_loss:.4f}\n'
                metric_str = self._generate_metric_str(metric_str)

            tqdm.write(self._log(metric_str))
            if self.metric.is_early_stop and e >= self.warm_epoch:
                tqdm.write(self._log("Early Stop!"))
                break
            else:
                self.metric.is_early_stop = False
        tqdm.write(self._log(self.metric.print_best_metrics()))

        self.metric.clear_metrics()
        all_loss = 0.0
        """
        write codes for test a epoch
        """
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        metric_str += f'loss: {all_loss:.4f}\n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self._log(metric_str))

        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
