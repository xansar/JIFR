#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LightGCNTrainer.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/26 13:15   zxx      1.0         None
"""
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import dgl
import torch
import numpy as np
from tqdm import tqdm, trange
import os


class LightGCNTrainer:
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

        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        # 设置log地址
        self.model_name = self.config['MODEL']['model_name']
        if not os.path.isdir('./log'):
            os.mkdir('./log')
        self.log_pth = self.config['TRAIN']['log_pth'] + str(self.random_seed) + f'_{self.model_name}.txt'

        # 打印config
        self.print_config()

        # 读取数据
        self.data_name = config['DATA']['data_name']
        self.dataset = dataset
        self.g = dataset[0]
        self.train_size = dataset.train_size
        self.val_size = dataset.val_size

        # 读取训练有关配置
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.neg_num = eval(config['DATA']['neg_num'])
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_step = eval(config['TRAIN']['eval_step'])

        # 读取metric配置
        self.metric = metric
        self.ks = eval(config['METRIC']['ks'])

        # 其他配置
        self.device = config['TRAIN']['device']
        self.to(self.device)

    def print_config(self):
        ## 用来打印config信息
        config_str = ''
        config_str += '=' * 10 + "Config" + '=' * 10 + '\n'
        for k, v in self.config.items():
            config_str += k + ': \n'
            for _k, _v in v.items():
                config_str += f'\t{_k}: {_v}\n'
        config_str += ('=' * 25 + '\n')
        tqdm.write(self.log(config_str, mode='w'))

    def to(self, device=None):
        # 整体迁移
        if device is None:
            self.model = self.model.to(self.config['TRAIN']['device'])
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
            self.config['TRAIN']['device'] = device

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            train_pos_g = inputs['train_pos_g']
            train_neg_g = inputs['train_neg_g']
            self.model.train()
            self.optimizer.zero_grad()
            pos_pred, neg_pred = self.model(
                train_pos_g,
                train_neg_g
            )
            loss = self.loss_func(pos_pred, neg_pred)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            return loss.item()
        elif mode == 'evaluate':
            with torch.no_grad():
                message_g = inputs['message_g']
                pred_g = inputs['pred_g']
                self.model.eval()
                # 先预测正样本，返回pred_g.num_edges() * 1
                user_embedding, item_embedding = self.model.get_embedding(message_g)
                pos_pred = self.model.predictor(pred_g, user_embedding, item_embedding)
                # 抽取self.neg_num个负样本进行预测
                neg_g = self.get_neg_graph(self.g, pred_g.edges()[0].repeat(self.neg_num), self.neg_num + 1)
                neg_pred = self.model.predictor(neg_g, user_embedding, item_embedding).reshape(-1, self.neg_num).cpu()
                loss = self.loss_func(pos_pred.reshape(-1), neg_pred[:, -1].to(self.device))
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def _generate_metric_str(self, metric_str):
        # 根据metric结果，生成文本
        for metric_name, k_dict in self.metric.metric_dict.items():
            for k, v in k_dict.items():
                metric_str += f'{metric_name}@{k}: {v["value"]:.4f}\t'
            metric_str += '\n'
        self.metric.clear_metrics()
        return metric_str

    def log(self, str_, mode='a'):
        # 将log写入文件
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def get_graphs(self):
        train_g = dgl.edge_subgraph(self.g, range(self.train_size), relabel_nodes=False)
        val_pred_g = dgl.edge_subgraph(self.g, range(self.train_size, self.train_size + self.val_size), relabel_nodes=False)
        val_g = dgl.edge_subgraph(self.g, range(self.train_size + self.val_size), relabel_nodes=False)
        test_pred_g = dgl.edge_subgraph(self.g, range(self.train_size + self.val_size, self.g.num_edges()), relabel_nodes=False)
        return train_g, val_pred_g, val_g, test_pred_g

    def get_neg_graph(self, graph, u, k=101):
        """
        为用户序列中的每个用户进行负采样，构建负采样图
        :param graph: 整图，一般是包含所有训练、测试样本的图
        :type graph: dgl.graph
        :param u: 待采样的用户id序列
        :type u: tensor
        :param k: 首次为每个用户负采样的数量，后续分配到每条正边的负边将从首次采样结果中产生
        :type k: int
        :return: 负采样图
        :rtype: dgl.graph
        """
        user_num = self.user_num
        neg_item_selected = torch.multinomial((1 - graph.adj().to_dense())[:user_num, user_num:], k)
        neg_item_idx = torch.randint(0, k, (1, u.shape[0])).reshape(-1)
        v = neg_item_selected[u, neg_item_idx].to(self.device)
        num_nodes = self.user_num + self.item_num
        return dgl.graph((u, v), num_nodes=num_nodes)

    def train(self):
        # 整体训练流程
        tqdm.write(self.log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        # 从左到右：训练图，用于验证的图，训练图+验证图，用于测试的图
        train_g, val_pred_g, val_g, test_pred_g = self.get_graphs()
        train_g = train_g.to(self.device)
        val_pred_g = val_pred_g.to(self.device)
        # val_g = val_g.to(self.device)
        test_pred_g = test_pred_g.to(self.device)
        train_neg_g = self.get_neg_graph(self.g, train_g.edges()[0])

        for e in range(1, epoch + 1):
            """
            write codes for train
            and return loss
            """
            loss = self.step(mode='train', train_pos_g=train_g, train_neg_g=train_neg_g)
            metric_str = f'Train Epoch: {e}\nLoss: {loss:.4f}\n'

            if e % self.eval_step == 0:
                # 在训练图上跑节点表示，在验证图上预测边的概率
                self.metric.clear_metrics()
                loss = self.step(mode='evaluate', message_g=train_g, pred_g=val_pred_g)
                self.metric.get_batch_metrics()
                metric_str += f'Evaluate Epoch: {e}\n'
                metric_str += f'loss: {loss:.4f}\n'
                metric_str = self._generate_metric_str(metric_str)
            tqdm.write(self.log(metric_str))
            if self.metric.is_early_stop:
                tqdm.write(self.log("Early Stop!"))
                break

        tqdm.write(self.log(self.metric.print_best_metrics()))

        self.metric.clear_metrics()
        # 在训练图上跑节点表示，在测试图上预测边的概率
        ## todo：确定好在什么图上跑message
        loss = self.step(mode='evaluate', message_g=train_g, pred_g=test_pred_g)
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        metric_str += f'loss: {loss:.4f}\n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self.log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)

