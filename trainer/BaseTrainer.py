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
from tqdm import tqdm
import os

"""
BaseTrainer主要用来写一些通用的函数，比如打印config之类
"""
class BaseTrainer:
    def __init__(self):
        pass

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
        for metric_name, k_dict in self.metric.metric_dict.items():
            for k, v in k_dict.items():
                metric_str += f'{metric_name}@{k}: {v["value"]:.4f}\t'
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
