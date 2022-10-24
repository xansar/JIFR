#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   trainer.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 21:30   zxx      1.0         None
"""
import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            loss_func: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_reg: torch.optim._LRScheduler,
            metric,
            train_rate_loader: torch.data.utils.Dataloader,
            train_link_loader: torch.data.utils.Dataloader,
            val_rate_loader: torch.data.utils.Dataloader,
            val_link_loader: torch.data.utils.Dataloader,
            test_rate_loader: torch.data.utils.Dataloader,
            test_link_loader: torch.data.utils.Dataloader,
            config,
            user_num=1597,
            item_num=24984
    ):
        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        self.log_pth = self.config['TRAIN']['log_pth'] + str(self.random_seed) + '_MutualRec.txt'
        self.print_config()

        self.train_rate_loader = train_rate_loader
        self.train_link_loader = train_link_loader
        self.val_rate_loader = val_rate_loader
        self.val_link_loader = val_link_loader
        self.test_rate_loader = test_rate_loader
        self.test_link_loader = test_link_loader

        self.user_num = user_num
        self.item_num = item_num
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_reg = lr_reg
        self.metric = metric
        self.data_name = config['DATA']['data_name']
        self.device = config['TRAIN']['device']
        self.eval_step = eval(config['TRAIN']['eval_step'])

        self.to(self.device)

    def print_config(self):
        config_str = ''
        config_str += '=' * 10 + "Config" + '=' * 10 + '\n'
        for k, v in self.config.items():
            config_str += k + ': \n'
            for _k, _v in v.items():
                config_str += f'\t{_k}: {_v}\n'
        config_str += ('=' * 25 + '\n')
        tqdm.write(self.log(config_str, mode='w'))

    def to(self, device=None):
        if device is None:
            self.model = self.model.to(self.config['TRAIN']['device'])
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
            self.config['TRAIN']['device'] = device

    def step(self, mode='train', **inputs):
        if mode == 'train':
            rate = inputs['rate']
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(inputs)
            loss = self.loss_func(pred, rate)
            loss.backward()
            self.optimizer.step()
            self.lr_reg.step()
            return loss.item()

        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                # ToDo: 负采样
                rate = inputs['rate']
                pred = self.model(inputs)
                loss = self.loss_func(pred, rate)
                loss.backward()
                self.optimizer.step()
                self.lr_reg.step()

                self.metric.compute_metrics(...)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def _compute_metric(self, metric_str):
        for metric_name, k_dict in self.metric.metric_dict.items():
            for k, v in k_dict.items():
                metric_str += f'{metric_name}@{k}: {v["value"]:.4f}\t'
            metric_str += '\n'
        self.metric.clear_metrics()
        return metric_str

    def log(self, str_, mode='a'):
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def train(self):
        tqdm.write(self.log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        self.metric.init_metrics()

        for e in range(1, epoch + 1):
            """
            write codes for train
            and return loss
            """
            all_loss = 0.0
            for idx, rate_data in enumerate(tqdm(self.train_rate_loader)):
                loss = self.step(
                    mode='train',
                    user=rate_data[:, 0].long(),
                    item = rate_data[:, 1].long(),
                    rate=rate_data[:, 2].float()
                )
                all_loss += loss
            all_loss /= idx
            metric_str = f'Train Epoch: {e}\nLoss: {all_loss:.4f}\n'

            if e % self.eval_step == 0:
                self.metric.clear_metrics()
                all_loss = 0.0
                for idx, rate_data in enumerate(tqdm(self.val_rate_loader)):
                    loss = self.step(
                        mode='evaluate',
                        user=rate_data[:, 0].long(),
                        item=rate_data[:, 1].long(),
                        rate=rate_data[:, 2].float()
                    )
                    all_loss += loss
                all_loss /= idx

                metric_str += f'Evaluate Epoch: {e}\n'
                metric_str += f'loss: {all_loss:.4f}\n'
                metric_str = self._compute_metric(metric_str)

            tqdm.write(self.log(metric_str))

        """
        write codes for test
        and return loss
        """
        metric_str = ''
        metric_str += f'Test Epoch: \n'
        metric_str += f'all loss: {loss:.4f}\nrate loss: {rate_loss:.4f}\nlink loss: {link_loss:.4f}\n'

        tqdm.write(self.log(self.metric.print_best_metrics()))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
