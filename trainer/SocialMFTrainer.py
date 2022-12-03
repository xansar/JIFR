#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   TrustSVDTrainer.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/4 21:19   zxx      1.0         None
"""

# import lib

import dgl
import torch
import numpy as np
from tqdm import tqdm, trange
import os

from .BaseTrainer import BaseTrainer

class SocialMFTrainer(BaseTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            loss_func: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler,
            metric,
            dataset,
            config,
    ):
        super(SocialMFTrainer, self).__init__(config)
        # 读取数据
        self.dataset = dataset
        self.g = dataset[0]
        self.train_rate_size = dataset.train_rate_size
        self.val_rate_size = dataset.val_rate_size
        self.train_link_size = dataset.train_link_size
        self.val_link_size = dataset.val_link_size

        # 读取训练有关配置
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.neg_num = eval(config['DATA']['neg_num'])
        self.train_neg_num = eval(config['DATA']['train_neg_num'])
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])

        # 读取metric配置
        self.metric = metric
        self.ks = eval(config['METRIC']['ks'])

        # 其他配置
        self.device = config['TRAIN']['device']
        self._to(self.device)

    def get_graphs(self):
        train_edges = {
            ('user', 'rate', 'item'): range(self.train_rate_size),
            ('item', 'rated-by', 'user'): range(self.train_rate_size),
            ('user', 'trusted-by', 'user'): range(self.train_link_size)
        }
        train_g = dgl.edge_subgraph(self.g, train_edges, relabel_nodes=False)

        val_edges = {
            ('user', 'rate', 'item'): range(self.train_rate_size, self.train_rate_size + self.val_rate_size),
            ('item', 'rated-by', 'user'): range(self.train_rate_size, self.train_rate_size + self.val_rate_size),
            ('user', 'trusted-by', 'user'): range(self.train_link_size, self.train_link_size + self.val_link_size)
        }
        val_pred_g = dgl.edge_subgraph(self.g, val_edges, relabel_nodes=False)

        test_edges = {
            ('user', 'rate', 'item'): range(self.train_rate_size + self.val_rate_size, self.g.num_edges(('user', 'rate', 'item'))),
            ('item', 'rated-by', 'user'): range(self.train_rate_size + self.val_rate_size, self.g.num_edges(('item', 'rated-by', 'user'))),
            ('user', 'trusted-by', 'user'): range(self.train_link_size + self.val_link_size, self.g.num_edges(('user', 'trusted-by', 'user')))
        }
        test_pred_g = dgl.edge_subgraph(self.g, test_edges, relabel_nodes=False)
        # val_g = dgl.edge_subgraph(self.g, range(self.train_rate_size + self.val_rate_size), relabel_nodes=False)

        return train_g, val_pred_g, test_pred_g

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            train_pos_g = inputs['train_pos_g']
            train_neg_g = self.construct_negative_graph(train_pos_g, self.train_neg_num, etype=('user', 'rate', 'item'))
            self.model.train()
            self.optimizer.zero_grad()
            pos_pred, neg_pred, reg_loss, link_loss = self.model(
                train_pos_g,
                train_pos_g,
                train_neg_g,
            )
            neg_pred = neg_pred.reshape(-1, self.train_neg_num)
            rate_loss = self.loss_func(pos_pred, neg_pred)
            loss = rate_loss + reg_loss + link_loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            return loss.item(), rate_loss.item(), reg_loss.item(), link_loss.item()
        elif mode == 'evaluate':
            with torch.no_grad():
                message_g = inputs['message_g']
                pred_g = inputs['pred_g']
                neg_g = self.construct_negative_graph(pred_g, self.neg_num, etype=('user', 'rate', 'item'))
                self.model.eval()
                pos_pred, neg_pred, reg_loss, link_loss = self.model(
                    message_g,
                    pred_g,
                    neg_g
                )
                neg_pred = neg_pred.reshape(-1, self.neg_num)
                rate_loss = self.loss_func(pos_pred, neg_pred)
                loss = rate_loss + reg_loss + link_loss
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                return loss.item(), rate_loss.item(), reg_loss.item(), link_loss.item()
        else:
            raise ValueError("Wrong Mode")

    def train(self):
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

            loss, rate_loss, reg_loss, link_loss = self.step(mode='train', train_pos_g=train_g)
            metric_str = f'Train Epoch: {e}\n' \
                         f'Loss: {loss:.4f}\t' \
                         f'Rate Loss: {rate_loss:.4f}\t' \
                         f'Reg Loss: {reg_loss:.4f}\t' \
                         f'Link Loss: {link_loss:.4f}\n'

            if e % self.eval_step == 0:
                # 在训练图上跑节点表示，在验证图上预测边的概率
                self.metric.clear_metrics()
                loss, rate_loss, reg_loss, link_loss = self.step(mode='evaluate', message_g=train_g, pred_g=val_pred_g)
                self.metric.get_batch_metrics()
                metric_str += f'Evaluate Epoch: {e}\n' \
                              f'Loss: {loss:.4f}\t' \
                              f'Rate Loss: {rate_loss:.4f}\t' \
                              f'Reg Loss: {reg_loss:.4f}\t' \
                              f'Link Loss: {link_loss:.4f}\n'
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
        # 在训练图上跑节点表示，在测试图上预测边的概率
        loss, rate_loss, reg_loss, link_loss = self.step(mode='evaluate', message_g=train_g, pred_g=test_pred_g)
        self.metric.get_batch_metrics()
        metric_str =  f'Test Epoch: \n' \
                      f'Loss: {loss:.4f}\t' \
                      f'Rate Loss: {rate_loss:.4f}\t' \
                      f'Reg Loss: {reg_loss:.4f}\t' \
                      f'Link Loss: {link_loss:.4f}\n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self._log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
