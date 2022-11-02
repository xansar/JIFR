#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MutualRecTrainer.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/30 16:26   zxx      1.0         None
"""
import dgl
import torch
import numpy as np
from tqdm import tqdm, trange
import os

from .BaseTrainer import BaseTrainer

class MutualRecTrainer(BaseTrainer):
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
        super(MutualRecTrainer, self).__init__()
        self.task = config['TRAIN']['task']

        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        # 设置log地址
        self.model_name = self.config['MODEL']['model_name']
        log_dir = self.config['TRAIN']['log_pth']
        if not os.path.isdir(os.path.join(log_dir, self.model_name)):
            os.mkdir(os.path.join(log_dir, self.model_name))
        self.log_pth = os.path.join(log_dir, self.model_name, f'{self.task}_{self.random_seed}_{self.model_name}.txt')
        # 设置保存地址
        save_dir = self.config['TRAIN']['save_pth']
        self.save_pth = os.path.join(save_dir, self.model_name, f'{self.task}_{self.random_seed}_{self.model_name}.pth')
        # 打印config
        self._print_config()

        # 读取数据
        self.data_name = config['DATA']['data_name']
        self.dataset = dataset
        self.g = dataset[0]
        self.user2history = self.dataset.user2history
        self.consumption_g = dataset.consumption_g
        self.social_g = dataset.social_g
        self.train_size = dataset.train_size
        self.val_size = dataset.val_size
        self.train_consumption_size = dataset.train_consumption_size
        self.val_consumption_size = dataset.val_consumption_size
        self.test_consumption_size = dataset.test_consumption_size

        # 读取训练有关配置
        self.pred_user_num = eval(config['MODEL']['pred_user_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.neg_num = eval(config['DATA']['neg_num'])
        self.num_sample = eval(config['MODEL']['num_sample'])
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
        train_g = dgl.edge_subgraph(self.g, range(self.train_size), relabel_nodes=False)
        val_pred_g = dgl.edge_subgraph(self.g, range(self.train_size, self.train_size + self.val_size), relabel_nodes=False)
        val_g = dgl.edge_subgraph(self.g, range(self.train_size + self.val_size), relabel_nodes=False)
        test_pred_g = dgl.edge_subgraph(self.g, range(self.train_size + self.val_size, self.g.num_edges()), relabel_nodes=False)
        return train_g, val_pred_g, val_g, test_pred_g

    def total_negative_sample(self, k, task='Rate'):
        # 为所有用户进行负样本抽样
        self.neg_samples = None
        for u in trange(self.pred_user_num, desc=f'{task} Neg Sampling', leave=True):
            # 如果是link任务，长度为total user num，rate任务是item num
            if task == 'Rate':
                # 当前u的互动列表，user和item的idx是从0开始计数的
                interacted_sample = self.user2history['user2item'][str(u)]
                mask = torch.ones(self.item_num)   # 0, item_num
                mask[interacted_sample] = 0
                # item id 计数从total user num开始
                cur_neg = torch.multinomial(mask, k, replacement=True).unsqueeze(0) + self.total_user_num
            else:
                interacted_sample = self.user2history['user2trust'][str(u)]
                mask = torch.ones(self.total_user_num)
                mask[interacted_sample] = 0
                # user 计数从0开始
                cur_neg = torch.multinomial(mask, k, replacement=True).unsqueeze(0)

            if self.neg_samples is None:
                self.neg_samples = cur_neg
            else:
                self.neg_samples = torch.vstack([self.neg_samples, cur_neg])

    def get_neg_graph(self, graph, u, k=100, mode='train', task='Rate'):
        """
        为用户序列中的每个用户进行负采样，构建负采样图
        :param mode: train：每个用户下的正样本对应的负样本不同，evaluate：每个用户下的正样本对应的负样本相同
        :type mode: str
        :param graph: 整图，一般是包含所有训练、测试样本的图
        :type graph: dgl.graph
        :param u: 待采样的用户id序列
        :type u: tensor
        :param k: 训练时为用户初次采样负样本的数量，之后正样本再从中抽取一个作为负样本
        :type k: int
        :return: 负采样图
        :rtype: dgl.graph
        """
        self.total_negative_sample(k, task)
        user_num = self.pred_user_num
        # 因为是二分图，只有（userid, itemid)这一部分是有交互的，user之间、item之间没有交互
        # 抽取的结果是user * k，即为每个用户抽一个k长度的负样本
        ## 这里一定注意，因为截取了user-item互动部分的矩阵，此时取出的idx下标是不对的，需要偏移user_num位
        # if task == 'Rate':
        #
        #     neg_sample_selected = \
        #         torch.multinomial((1 - graph.adj().to_dense())[:user_num, self.total_user_num:], k, replacement=True)\
        #         + self.total_user_num
        #     num_nodes = self.total_user_num + self.item_num
        # else:
        #     neg_sample_selected = \
        #         torch.multinomial((1 - graph.adj().to_dense())[:user_num, :], k, replacement=True)
        num_nodes = self.total_user_num + self.item_num

        # 测试时用户拥有的正样本对应同一组负样本
        if mode == 'evaluate' or mode == 'test':
            v = self.neg_samples[u].t().reshape(-1).to(self.device)
            u = u.repeat(k)
        else:
            # 为每个正样本从这k个中抽取一个负样本
            neg_sample_idx = torch.randint(0, k, (1, u.shape[0])).reshape(-1)
            v = self.neg_samples[u, neg_sample_idx].to(self.device)

        return dgl.graph((u, v), num_nodes=num_nodes)

    def get_social_neighbour_network(self, g):
        pred_user = range(self.pred_user_num)
        total_user = range(self.total_user_num)
        # 采样社交邻接网络，同时计算拉普拉斯矩阵特征值
        ## 采样
        social_network = dgl.node_subgraph(g, total_user)
        social_neighbout_nerwork = dgl.sampling.sample_neighbors(
            social_network,
            pred_user,
            fanout=self.num_sample,
            replace=False,
            edge_dir='out')
        social_neighbout_nerwork = dgl.to_bidirected(social_neighbout_nerwork.to('cpu'))
        ## 计算特征值
        laplacian_lambda_max = dgl.laplacian_lambda_max(social_neighbout_nerwork)
        laplacian_lambda_max = torch.tensor(laplacian_lambda_max, dtype=torch.float32)
        return social_neighbout_nerwork.to(self.device), laplacian_lambda_max.to(self.device)

    def get_graphs_for_model_forward(self, g, consumption_size):
        pred_user = range(self.pred_user_num)
        # item influence, 因为consumption_g包好所有的pred user和item节点，只需要将这两部分的embedding拼起来即可，不需要对应id
        consumption_g = dgl.edge_subgraph(g, edges=range(consumption_size))  # 要注意节点id的对应
        ## 顶层图
        consumption_neighbour_g = dgl.sampling.sample_neighbors(
            g=consumption_g,
            nodes=pred_user,  # 对所有的用户都采item的邻居
            fanout=self.num_sample,
            replace=False,
            edge_dir='out'  # 都是user指向item
        )
        reverse_consumption_neighbour_g = dgl.reverse(consumption_neighbour_g)

        ## 底层图
        sampled_item = torch.unique(consumption_neighbour_g.edges()[1])  # consumption_graph的id
        user2item_g = dgl.sampling.sample_neighbors(
            g=consumption_g,
            nodes=sampled_item,
            fanout=self.num_sample,
            replace=False,
            edge_dir='in'
        )

        # social influence
        ## 从整图中抽取社交网络
        pred_user_social_sub_g = dgl.node_subgraph(g, nodes=pred_user)

        social_neighbour_g = dgl.sampling.sample_neighbors(
            g=pred_user_social_sub_g,
            nodes=pred_user,
            fanout=self.num_sample,
            replace=False,
            edge_dir='in'  # 其实in和out采的边的分布一样，只是边的方向不一样
            # 使用in，那么边(u, v)中v就是完整的nodes列表
        )
        social_neighbour_g = dgl.to_bidirected(social_neighbour_g.to('cpu'))
        ## user列表
        sampled_user = torch.unique(social_neighbour_g.edges()[0]).tolist()
        ## item列表
        item_id_range = list(range(self.pred_user_num, consumption_g.num_nodes()))
        social_consumption_neighbour_g = dgl.sampling.sample_neighbors(
            g=consumption_g,  # pred user 部分id与social neigh一样，item部分多出来的就是item_id_range的范围
            nodes=pred_user,  # 对所有的用户都采item的邻居
            fanout=self.num_sample,
            replace=False,
            edge_dir='out'  # 都是user指向item
        )
        # 这里又重新编号了
        item2user_g = dgl.node_subgraph(social_consumption_neighbour_g, nodes=sampled_user + item_id_range)
        item2user_g = dgl.reverse(item2user_g)
        social_neighbout_nerwork, laplacian_lambda_max = self.get_social_neighbour_network(g)
        graphs = {
            'g': g.to(self.device),
            'user2item_g': user2item_g.to(self.device),
            'reverse_consumption_neighbour_g': reverse_consumption_neighbour_g.to(self.device),
            'item2user_g': item2user_g.to(self.device),
            'social_neighbour_g': social_neighbour_g.to(self.device),
            'social_neighbour_network': social_neighbout_nerwork
        }
        return graphs, laplacian_lambda_max

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if mode == 'train':
            train_pos_g = inputs['train_pos_g']
            graphs, laplacian_lambda_max = self.get_graphs_for_model_forward(train_pos_g, self.train_consumption_size)
            # train_neg_g = self.get_neg_graph(self.g, train_pos_g.edges()[0], self.neg_num, mode=mode)

            self.model.train()
            self.optimizer.zero_grad()
            h_new_P, h_new_S = self.model(graphs, laplacian_lambda_max)
            # BPRloss计算，需要从原始图中采一正一负两张图，并且由于embedding的问题，最好这两张图的节点id与原始图一致
            ## rate
            pos_rate_g = dgl.edge_subgraph(train_pos_g, edges=range(self.train_consumption_size), relabel_nodes=False)
            neg_rate_g = self.get_neg_graph(self.consumption_g, pos_rate_g.edges()[0], self.neg_num, mode, 'Rate')
            pos_rate_pred = self.model.predict(pos_rate_g, h_new_P)
            neg_rate_pred = self.model.predict(neg_rate_g, h_new_P)
            rate_loss = self.loss_func(pos_rate_pred, neg_rate_pred)

            ## link
            pos_link_g = dgl.edge_subgraph(train_pos_g,
                                           edges=range(self.train_consumption_size, train_pos_g.num_edges()),
                                           relabel_nodes=False)
            neg_link_g = self.get_neg_graph(self.social_g, pos_link_g.edges()[0], self.neg_num, mode, 'Link')
            pos_link_pred = self.model.predict(pos_link_g, h_new_S)
            neg_link_pred = self.model.predict(neg_link_g, h_new_S)
            link_loss = self.loss_func(pos_link_pred, neg_link_pred)

            loss = rate_loss + link_loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            return loss.item(), rate_loss.item(), link_loss.item()
        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                message_g = inputs['message_g']
                pred_g = inputs['pred_g']
                graphs, laplacian_lambda_max = self.get_graphs_for_model_forward(message_g,
                                                                                 self.train_consumption_size)
                self.model.eval()
                # 先预测正样本，返回pred_g.num_edges() * 1
                h_new_P, h_new_S = self.model(graphs, laplacian_lambda_max)
                ## rate
                consumption_size = self.val_consumption_size if mode == 'evaluate' else self.test_consumption_size
                pos_rate_g = dgl.edge_subgraph(pred_g, edges=range(consumption_size),
                                               relabel_nodes=False)
                neg_rate_g = self.get_neg_graph(self.consumption_g, pos_rate_g.edges()[0], self.neg_num, mode, 'Rate')
                pos_rate_pred = self.model.predict(pos_rate_g, h_new_P)
                neg_rate_pred = self.model.predict(neg_rate_g, h_new_P).reshape(self.neg_num, -1).t()
                rate_loss = self.loss_func(pos_rate_pred.reshape(-1), torch.mean(neg_rate_pred, dim=1))
                self.metric.compute_metrics(pos_rate_pred.cpu(), neg_rate_pred.cpu(), task='Rate')

                ## link
                consumption_size = self.val_consumption_size if mode == 'evaluate' else self.test_consumption_size
                pos_link_g = dgl.edge_subgraph(pred_g, edges=range(consumption_size, pred_g.num_edges()),
                                            relabel_nodes=False)
                neg_link_g = self.get_neg_graph(self.social_g, pos_link_g.edges()[0], self.neg_num, mode, 'Link')
                pos_link_pred = self.model.predict(pos_link_g, h_new_S)
                neg_link_pred = self.model.predict(neg_link_g, h_new_S).reshape(self.neg_num, -1).t()
                link_loss = self.loss_func(pos_link_pred.reshape(-1), torch.mean(neg_link_pred, dim=1))
                self.metric.compute_metrics(pos_link_pred.cpu(), neg_link_pred.cpu(), task='Link')

                loss = rate_loss + link_loss
                return loss.item(), rate_loss.item(), link_loss.item()
        else:
            raise ValueError("Wrong Mode")

    def train(self):
        # 整体训练流程
        tqdm.write(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        # 从左到右：训练图，用于验证的图，训练图+验证图，用于测试的图
        train_g, val_pred_g, val_g, test_pred_g = self.get_graphs()
        train_g = train_g.to(self.device)
        val_pred_g = val_pred_g.to(self.device)
        # val_g = val_g.to(self.device)
        test_pred_g = test_pred_g.to(self.device)

        for e in range(1, epoch + 1):
            """
            write codes for train
            and return loss
            """
            loss, rate_loss, link_loss = self.step(mode='train', train_pos_g=train_g)
            metric_str = f'Train Epoch: {e}\n' \
                         f'Loss: {loss:.4f}\t' \
                         f'Rate Loss: {rate_loss:.4f}\t' \
                         f'Link Loss: {link_loss:.4f}\n'

            if e % self.eval_step == 0:
                # 在训练图上跑节点表示，在验证图上预测边的概率
                self.metric.clear_metrics()
                loss, rate_loss, link_loss = self.step(mode='evaluate', message_g=train_g, pred_g=val_pred_g)
                self.metric.get_batch_metrics()
                metric_str += f'Evaluate Epoch: {e}\n'
                metric_str += f'Loss: {loss:.4f}\t' \
                              f'Rate Loss: {rate_loss:.4f}\t' \
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
        loss, rate_loss, link_loss = self.step(mode='test', message_g=train_g, pred_g=test_pred_g)
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        metric_str += f'Loss: {loss:.4f}\t' \
                      f'Rate Loss: {rate_loss:.4f}\t' \
                      f'Link Loss: {link_loss:.4f}\n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self._log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
