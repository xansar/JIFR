#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ExtendedEpinionsMF.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 21:30   zxx      1.0         None
"""
import torch
import numpy as np
from tqdm import tqdm, trange
import os
import json

from .BaseTrainer import BaseTrainer

class MFTrainer(BaseTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            loss_func: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler,
            metric,
            dataloader_dict,
            config,
    ):
        super(MFTrainer, self).__init__()
        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        self.model_name = self.config['MODEL']['model_name']
        self.task = self.config['TRAIN']['task']
        self.log_dir = self.config['TRAIN']['log_pth']
        if not os.path.isdir(os.path.join(self.log_dir, self.model_name)):
            os.mkdir(os.path.join(self.log_dir, self.model_name))
        self.log_pth = os.path.join(self.log_dir, self.model_name, str(self.random_seed) + f'_{self.model_name}.txt')
        # 设置保存地址
        save_dir = self.config['TRAIN']['save_pth']
        self.save_pth = os.path.join(save_dir, self.model_name, f'{self.random_seed}_{self.model_name}.pth')
        # 打印config
        self._print_config()

        self.train_loader = dataloader_dict['train']
        # self.train_link_loader = train_link_loader
        self.val_loader = dataloader_dict['val']
        # self.val_link_loader = val_link_loader
        self.test_loader = dataloader_dict['test']
        # self.test_link_loader = test_link_loader

        self.user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.metric = metric
        self.user2history = self.train_loader.dataset.user2history
        self.neg_num = eval(config['DATA']['neg_num'])

        self.data_name = config['DATA']['data_name']
        self.device = config['TRAIN']['device']
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])
        self.ks = eval(config['METRIC']['ks'])

        self._to(self.device)

    def step(self, mode='train', **inputs):
        if mode == 'train':
            # rate: u-user, v-item; link: u-user, v-trusted user
            u = inputs['u'].to(self.device)
            v = inputs['v'].to(self.device)
            weight = inputs['weight'].to(self.device)

            self.model.train()
            self.optimizer.zero_grad()

            neg_sample = self.get_negative_sample(u, k=1).reshape(-1)
            pos_pred = self.model(
                u=u,
                v=v
            )
            neg_pred = self.model(
                u=u,
                v=neg_sample
            )
            loss = self.loss_func(pos_pred, weight)
            neg_loss = self.loss_func(neg_pred, torch.zeros_like(weight, device=self.device))
            loss = loss * 0.5 + neg_loss * 0.5
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            return loss.item()

        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                # rate: u-user, v-item; link: u-user, v-trusted user
                u = inputs['u'].to(self.device)
                v = inputs['v'].to(self.device)
                weight = inputs['weight'].to(self.device)
                # 一定注意转置和reshape的顺序
                neg_sample = self.get_negative_sample(u, k=self.neg_num).t().reshape(-1)
                pos_pred = self.model(
                    u=u,
                    v=v
                )
                # 一定注意转置和reshape的顺序
                neg_pred = self.model(
                    u=u.repeat(self.neg_num),
                    v=neg_sample
                ).reshape(self.neg_num, -1).t()
                loss = self.loss_func(pos_pred, weight)
                neg_loss = self.loss_func(torch.mean(neg_pred, dim=1), torch.zeros_like(weight, device=self.device))
                loss = loss * 0.5 + neg_loss * 0.5
                pos_pred = pos_pred.cpu().reshape(-1, 1)
                neg_pred = neg_pred.cpu()
                self.metric.compute_metrics(pos_pred, neg_pred)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def total_negative_sample(self, k):
        # 为所有用户进行负样本抽样
        self.neg_samples = None
        for u in trange(self.user_num, desc='Neg Sampling', leave=True):
            # 当前u的互动列表，user和item的idx是从0开始计数的
            interacted_sample = self.user2history[str(u)]
            # 如果是link任务，长度为total user num，rate任务是item num
            if self.task == 'Rate':
                mask = torch.ones(self.item_num)   # 0, item_num
            else:
                mask = torch.ones(self.total_user_num)
            mask[interacted_sample] = 0
            # 这是在全局抽100个样本，这里不用加id偏移的原因是模型内部embedding是分开的，分别从0开始计数
            cur_neg = torch.multinomial(mask, k, replacement=True).unsqueeze(0)
            if self.neg_samples is None:
                self.neg_samples = cur_neg
            else:
                self.neg_samples = torch.vstack([self.neg_samples, cur_neg])

    def get_negative_sample(self, user: torch.Tensor, k):
        return self.neg_samples[user, :k]
        # high = self.neg_samples.shape[1]
        # # 生成一个跟user同等长度，宽度为k的随机idx矩阵
        # idx = torch.randint(0, high, (user.shape[0], k))
        # return self.neg_samples[user].gather(1, idx)

    def train(self):
        tqdm.write(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])

        for e in range(1, epoch + 1):
            all_loss = 0.0
            self.total_negative_sample(k=self.neg_num)
            for idx, data in enumerate(tqdm(self.train_loader)):
                loss = self.step(
                    mode='train',
                    u=data[:, 0].long(),
                    v=data[:, 1].long(),
                    weight=data[:, 2].float()
                )
                all_loss += loss
            all_loss /= idx
            metric_str = f'Train Epoch: {e}\nLoss: {all_loss:.4f}\n'

            if e % self.eval_step == 0:
                self.total_negative_sample(k=self.neg_num)
                self.metric.clear_metrics()
                all_loss = 0.0
                for idx, data in enumerate(tqdm(self.val_loader)):
                    loss = self.step(
                        mode='evaluate',
                        u=data[:, 0].long(),
                        v=data[:, 1].long(),
                        weight=data[:, 2].float(),
                    )
                    all_loss += loss
                all_loss /= idx
                self.metric.get_batch_metrics()
                metric_str += f'Evaluate Epoch: {e}\n'
                metric_str += f'loss: {all_loss:.4f}\n'
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
        tqdm.write(self._log(self.metric.print_best_metrics()))

        # 开始测试
        # 加载最优模型
        self._load_model(self.save_pth)
        self.metric.clear_metrics()
        all_loss = 0.0
        for idx, data in enumerate(tqdm(self.test_loader)):
            loss = self.step(
                mode='evaluate',
                u=data[:, 0].long(),
                v=data[:, 1].long(),
                weight=data[:, 2].float(),
            )
            all_loss += loss
        all_loss /= idx
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        metric_str += f'loss: {all_loss:.4f}\n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self._log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
