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

from numba.typed import List
from numba import njit
from numba.types import ListType, int64, int32, Array

from tqdm import tqdm, trange
import os
import json
import dgl
import time

"""
BaseTrainer主要用来写一些通用的函数，比如打印config之类
"""

# 使用numba加速负采样
@njit(Array(int64, 1, 'C')(ListType(int64,), ListType(ListType(int64,),), int64, int64))
def neg_sampling(u_lst, pos_lst, neg_num, total_num):
    lst = []
    lst.append(0)
    lst.pop()
    for u in u_lst:
        history = pos_lst[u]
        for _ in range(neg_num):
            j = np.random.randint(total_num)
            while j in history:
                j = np.random.randint(total_num)
            lst.append(j)
    return np.array(lst)

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

        self.step_per_epoch = eval(self.config['TRAIN']['step_per_epoch'])
        self.get_history_lst()

    def get_history_lst(self):
        # 获取用户历史行为列表
        with open(f'./data/{self.data_name}/splited_data/user2v.json', 'r', encoding='utf-8') as f:
            user2v = json.load(f)

        user2item = user2v['user2item']
        user2trust = user2v['user2trust']
        self.u_lst = List()
        self.item_lst = List()
        self.trust_lst = List()

        assert user2trust.keys() == user2item.keys()

        if self.task == 'Rate' or self.task == 'Joint':
            for u, v in user2item.items():
                self.u_lst.append(int(u))
                self.item_lst.append(List(v))

        if self.task == 'Link' or self.task == 'Joint':
            for u, v in user2trust.items():
                if self.task == 'Link':
                    self.u_lst.append(int(u))
                self.trust_lst.append(List(v))

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
        # 负采样，按用户进行
        utype, _, vtype = etype
        src, dst = graph.edges(etype=etype)
        neg_src = src.repeat_interleave(k)

        history_lst = self.trust_lst if vtype == utype else self.item_lst
        total_num = self.total_user_num if vtype == utype else self.item_num

        neg_samples = torch.from_numpy(neg_sampling(self.u_lst, history_lst, k, total_num)).reshape(-1, k)
        neg_dst = neg_samples[src].reshape(-1).to(self.device)

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

    def _train(self, loss_name):
        # 整体训练流程
        tqdm.write(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        # 从左到右：训练图，用于验证的图，用于测试的图
        train_g, val_pred_g, test_pred_g = self.get_graphs()
        train_g = train_g.to(self.device)
        val_pred_g = val_pred_g.to(self.device)
        test_pred_g = test_pred_g.to(self.device)
        for e in range(1, epoch + 1):
            """
            write codes for train
            and return loss
            """
            all_loss_lst = [0.0 for _ in range(len(loss_name))]
            for _ in trange(self.step_per_epoch):
                loss_lst = self.step(mode='train', train_pos_g=train_g)
                for j in range(len(loss_name)):
                    if len(loss_name) == 1:
                        loss_lst = [loss_lst]
                    all_loss_lst[j] += loss_lst[j]
            metric_str = f'Train Epoch: {e}\n'
            for j in range(len(loss_name)):
                all_loss_lst[j] /= self.step_per_epoch
                metric_str += f'{loss_name[j]}: {all_loss_lst[j]:.4f}\t'
            tqdm.write(self._log(metric_str))

            if e % self.eval_step == 0:
                # 在训练图上跑节点表示，在验证图上预测边的概率
                self.metric.clear_metrics()
                loss_lst = self.step(mode='evaluate', message_g=train_g, pred_g=val_pred_g)
                self.metric.get_batch_metrics()
                metric_str = f'Evaluate Epoch: {e}\n'
                for j in range(len(loss_name)):
                    metric_str += f'{loss_name[j]}: {loss_lst[j]:.4f}\t'
                metric_str += '\n'
                metric_str = self._generate_metric_str(metric_str)
                tqdm.write(self._log(metric_str))

                # 保存最好模型
                if self.metric.is_save:
                    self._save_model(self.save_pth)
                    self.metric.is_save = False
                # 是否早停
                if self.metric.is_early_stop and e >= self.warm_epoch:
                    tqdm.write(self._log("Early Stop!"))
                    break
                else:
                    self.metric.is_early_stop = False
            self.lr_scheduler.step()
        tqdm.write(self._log(self.metric.print_best_metrics()))

        # 开始测试
        # 加载最优模型
        self._load_model(self.save_pth)
        self.metric.clear_metrics()
        # 在训练图上跑节点表示，在测试图上预测边的概率
        loss_lst = self.step(mode='evaluate', message_g=train_g, pred_g=test_pred_g)
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        for j in range(len(loss_name)):
            metric_str += f'{loss_name[j]}: {loss_lst[j]:.4f}\t'
        metric_str += '\n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self._log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
