#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Node2VecTrainer.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/4 21:19   zxx      1.0         None
"""

# import lib

import dgl
from dgl.sampling import node2vec_random_walk
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
import os

from .BaseTrainer import BaseTrainer

class Node2VecTrainer(BaseTrainer):
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
        super(Node2VecTrainer, self).__init__(config)
        # 读取数据
        self.dataset = dataset
        self.g = dataset[0]
        self.train_size = dataset.train_size
        self.val_size = dataset.val_size

        # 读取训练有关配置
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.neg_num = eval(config['DATA']['neg_num'])
        self.train_neg_num = eval(config['DATA']['train_neg_num'])
        self.batch_size = eval(config['DATA']['batch_size'])
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])
        self.p = eval(config['MODEL']['p'])
        self.q = eval(config['MODEL']['q'])
        self.num_walks = eval(config['MODEL']['num_walks'])
        self.walk_length = eval(config['MODEL']['walk_length'])
        self.window_size = eval(config['MODEL']['window_size'])

        # 读取metric配置
        self.metric = metric
        self.ks = eval(config['METRIC']['ks'])

        # 其他配置
        self.device = config['TRAIN']['device']
        self._to(self.device)

    def get_graphs(self):
        train_edges = {
            ('user', 'trust', 'user'): range(self.train_size),
            ('user', 'trusted-by', 'user'): range(self.train_size)
        }
        train_g = dgl.edge_subgraph(self.g, train_edges, relabel_nodes=False)

        val_edges = {
            ('user', 'trust', 'user'): range(self.train_size, self.train_size + self.val_size),
            ('user', 'trusted-by', 'user'): range(self.train_size, self.train_size + self.val_size)
        }
        val_pred_g = dgl.edge_subgraph(self.g, val_edges, relabel_nodes=False)

        test_edges = {
            ('user', 'trust', 'user'): range(self.train_size + self.val_size, self.g.num_edges(('user', 'trusted-by', 'user'))),
            ('user', 'trusted-by', 'user'): range(self.train_size + self.val_size, self.g.num_edges(('user', 'trusted-by', 'user')))
        }
        test_pred_g = dgl.edge_subgraph(self.g, test_edges, relabel_nodes=False)
        # val_g = dgl.edge_subgraph(self.g, range(self.train_rate_size + self.val_size), relabel_nodes=False)

        return train_g, val_pred_g, test_pred_g

    def loader(self, batch_size):
        """
        Parameters
        ----------
        batch_size: int
            batch size
        Returns
        -------
        DataLoader
            Node2vec training data loader
        """
        return DataLoader(
            torch.arange(self.user_num),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.sample,
        )

    def sample(self, batch):
        """
        Generate positive and negative samples.
        Positive samples are generated from random walk
        Negative samples are generated from random sampling
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch = batch.repeat(self.num_walks)
        # positive
        pos_traces = node2vec_random_walk(
            self.train_g, batch, self.p, self.q, self.walk_length
        )
        pos_traces = pos_traces.unfold(1, self.window_size, 1)  # rolling window
        pos_traces = pos_traces.contiguous().view(-1, self.window_size)

        # negative
        neg_batch = batch.repeat(self.train_neg_num)
        neg_traces = torch.randint(
            self.user_num, (neg_batch.size(0), self.walk_length)
        )
        neg_traces = torch.cat([neg_batch.view(-1, 1), neg_traces], dim=-1)
        neg_traces = neg_traces.unfold(1, self.window_size, 1)  # rolling window
        neg_traces = neg_traces.contiguous().view(-1, self.window_size)

        return pos_traces, neg_traces

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            pos_traces = inputs['pos_traces'].to(self.device)
            neg_traces = inputs['neg_traces'].to(self.device)
            self.model.train()
            self.optimizer.zero_grad()
            pos_loss, neg_loss = self.model.loss(
                pos_traces,
                neg_traces
            )
            loss = pos_loss + neg_loss
            loss.backward()
            self.optimizer.step()
            return loss.item()
        elif mode == 'evaluate':
            with torch.no_grad():
                message_g = inputs['message_g']
                pred_g = inputs['pred_g']
                neg_g = self.construct_negative_graph(pred_g, self.neg_num, etype=('user', 'trust', 'user'))
                self.model.eval()
                pos_pred, neg_pred = self.model.evaluate(
                    message_g,
                    pred_g,
                    neg_g,
                )
                neg_pred = neg_pred.reshape(-1, self.neg_num)
                loss = self.loss_func(pos_pred, neg_pred)
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def train(self):
        loss_name = ['Loss']
        self._train(loss_name)

    def _train(self, loss_name, side_info: dict=None):
        bar_range = trange if self.step_per_epoch > 10 else lambda x: range(x)
        # 整体训练流程
        tqdm.write(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        # 从左到右：训练图，用于验证的图，用于测试的图
        train_g, val_pred_g, test_pred_g = self.get_graphs()
        self.train_g = train_g.to(self.device)
        self.val_pred_g = val_pred_g.to(self.device)
        self.test_pred_g = test_pred_g.to(self.device)

        loader = self.loader(self.batch_size)

        # 分bin测试用
        if self.bin_sep_lst is not None:
            train_e_id_lst, val_e_id_lst, test_e_id_lst = self.get_bins_eid_lst()
        side_info = self.get_side_info()

        # # drop graph
        # drop_edger = dgl.DropEdge()
        for e in range(1, epoch + 1):
            """
            write codes for train
            and return loss
            """
            self.cur_e = e

            # 分箱看histgram
            if self.bin_sep_lst is not None and self.is_visulized == True:
                self.bins_id_lst = train_e_id_lst

            all_loss_lst = [0.0 for _ in range(len(loss_name))]
            for i, (pos_traces, neg_traces) in tqdm(enumerate(loader), total=int(self.user_num / self.batch_size) + 1):
                loss_lst = self.step(mode='train', pos_traces=pos_traces, neg_traces=neg_traces, cur_step=i)
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
                if self.bin_sep_lst is not None:
                    self.bins_id_lst = val_e_id_lst
                # 在训练图上跑节点表示，在验证图上预测边的概率
                self.metric.clear_metrics()
                val_loss_lst = self.step(mode='evaluate', message_g=self.train_g, pred_g=self.val_pred_g, side_info=side_info)
                self.metric.get_batch_metrics()
                metric_str = f'Evaluate Epoch: {e}\n'
                for j in range(len(loss_name)):
                    if len(loss_name) == 1:
                        val_loss_lst = [val_loss_lst]
                    metric_str += f'{loss_name[j]}: {val_loss_lst[j]:.4f}\t'
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

                if self.is_visulized:
                    for j in range(len(loss_name)):
                        self.writer.add_scalar(f'loss/Train_{loss_name[j]}', all_loss_lst[j], e)
                        self.writer.add_scalar(f'loss/Val_{loss_name[j]}', val_loss_lst[j], e)

            self.lr_scheduler.step()
        tqdm.write(self._log(self.metric.print_best_metrics()))

        # 开始测试
        # 加载最优模型
        self._load_model(self.save_pth)
        self.metric.clear_metrics()

        # 分bin
        if self.bin_sep_lst is not None:
            self.metric.bins_id_lst = test_e_id_lst
            self.bins_id_lst = test_e_id_lst

        # 在训练图上跑节点表示，在测试图上预测边的概率
        loss_lst = self.step(mode='test', message_g=self.train_g, pred_g=self.test_pred_g, side_info=side_info)
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        for j in range(len(loss_name)):
            if len(loss_name) == 1:
                loss_lst = [loss_lst]
            metric_str += f'{loss_name[j]}: {loss_lst[j]:.4f}\t'
        metric_str += '\n'
        metric_str = self._generate_metric_str(metric_str, is_val=False)
        tqdm.write(self._log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
