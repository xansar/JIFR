#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   BaseTrainer.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/28 17:15   zxx      1.0         None
"""


from dataset import *
from metric import *
from model import *
from trainer.utils import *

import dgl
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import optuna

from tqdm import tqdm, trange
from tabulate import tabulate

import time
import os
import json
import copy

cf_model = ['LightGCN', 'MF', 'NCL']
social_model = ['MutualRec', 'DiffnetPP', 'GraphRec',
                'NJBP', 'SocialLGN', 'FLGN', 'SSCL', 'FNCL', 'CrossFire']
directed_social_model = ['TrustSVD', 'SVDPP', 'Sorec', 'SocialMF']
link_model = ['AA', 'Node2Vec']



"""
BaseTrainer主要用来写一些通用的函数，比如打印config之类
"""

class BaseTrainer:
    def __init__(self, config, trial):
        # 超参搜索
        self.trial = trial

        self.task = config['TRAIN']['task']
        self.is_visulized = config['VISUALIZED']
        self.is_log = config['LOG']
        self.print_info = tqdm.write if self.is_log else lambda x: None

        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        # 设置log地址
        self.model_name = self.config['MODEL']['model_name']
        self.data_name = config['DATA']['data_name']
        log_dir = self.config['TRAIN']['log_pth']
        if not os.path.isdir(os.path.join(log_dir, self.model_name, self.data_name)):
            os.makedirs(os.path.join(log_dir, self.model_name, self.data_name))
        trial_text = 'HyperParamsSearch_' if self.trial is not None else ''
        self.log_pth = os.path.join(log_dir, self.model_name, self.data_name, f'{trial_text}{self.task}_{self.random_seed}_{self.model_name}.txt')

        # 设置保存地址
        save_dir = self.config['TRAIN']['save_pth']
        if not os.path.isdir(os.path.join(save_dir, self.model_name, self.data_name)):
            os.makedirs(os.path.join(save_dir, self.model_name, self.data_name))
        self.save_pth = os.path.join(save_dir, self.model_name, self.data_name, f'{trial_text}{self.task}_{self.random_seed}_{self.model_name}.pth')

        # 早停
        self.early_stopper = EarlyStopper(
            patience=eval(config['OPTIM']['patience']),
            minimum_impro=eval(config['OPTIM']['minimum_impro']),
        )

        # 打印config
        self._print_config()
        if self.trial is not None:
            self.bin_sep_lst = None
        else:
            self.bin_sep_lst = eval(config['METRIC'].get('bin_sep_lst', 'None'))


        if self.is_visulized:
            tensor_board_dir = os.path.join(
                log_dir, self.model_name, self.data_name, f'{self.task}_{self.random_seed}_{self.model_name}')
            self.writer = SummaryWriter(tensor_board_dir)
            self.print_info(self._log(f'tensor_board_pth: {tensor_board_dir}'))
            self.vis_cnt = 1

        self.model_name = config['MODEL']['model_name']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        # self.neg_num = eval(config['DATA']['neg_num'])
        self.train_neg_num = eval(config['DATA']['train_neg_num'])
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])
        self.ks = eval(config['METRIC']['ks'])
        self.device = torch.device(config['TRAIN']['device'])

        self._build()
        self._to(self.device)

    def _build(self):
        self._build_metric()
        self._build_data()
        self._build_model()
        self._build_optimizer()

    def _build_model(self):
        # model
        self.model = eval(self.model_name + 'Model')(self.config, self.etypes)

        # loss func
        loss_name = self.config['LOSS']['loss_name']
        self.loss_func = eval(loss_name)(reduction='mean')

    def _build_optimizer(self):
        config = self.config
        # optimizer
        optimizer_name = 'torch.optim.' + config['OPTIM']['optimizer']
        if 'mlp_weight_decay' in config['OPTIM'].keys():
            optimizer_grouped_params = [
                {'params': [p for n, p in self.model.named_parameters() if 'embeds' in n],
                 'lr': eval(config['OPTIM']['embedding_learning_rate']),
                 'weight_decay': eval(config['OPTIM']['embedding_weight_decay'])
                 },
                {'params': [p for n, p in self.model.named_parameters() if 'embeds' not in n],
                 'lr': eval(config['OPTIM']['mlp_learning_rate']),
                 'weight_decay': eval(config['OPTIM']['mlp_weight_decay'])
                 }
            ]
            self.optimizer = eval(optimizer_name)(params=optimizer_grouped_params)
        else:
            lr = eval(config['OPTIM']['embedding_learning_rate'])
            weight_decay = eval(config['OPTIM']['embedding_weight_decay'])
            self.optimizer = eval(optimizer_name)(lr=lr, params=self.model.parameters(), weight_decay=weight_decay)

        # lr_scheduler_name = 'torch.optim.lr_scheduler.' + config['OPTIM']['lr_scheduler']
        # T_0 = eval(config['OPTIM']['T_0'])  # 学习率第一次重启的epoch数
        # T_mult = eval(config['OPTIM']['T_mult'])  # 学习率衰减epoch数变化倍率
        # self.lr_scheduler = eval(lr_scheduler_name)(self.optimizer, T_0=T_0, T_mult=T_mult, verbose=self.is_log)

    def _build_metric(self):
        # metric
        self.metric = BaseMetric(self.config, self.bin_sep_lst)
        self.save_metric = self.metric.save_metric
        self.indicator_k = self.metric.indicator_k
        self.indicator_task = self.metric.indicator_task

    def _build_data(self):
        config = self.config
        # get data loader
        dataset = DGLRecDataset(config)
        self.etypes = dataset[0].etypes
        self.g = dataset[0]
        self.g = self._get_model_specific_etype_graph(self.g)

        # self.train_rate_size = dataset.train_rate_size
        # self.val_rate_size = dataset.val_rate_size
        # self.train_link_size = dataset.train_link_size
        # self.val_link_size = dataset.val_link_size

        train_batch_size = eval(config['DATA']['train_batch_size'])
        val_batch_size = eval(config['DATA']['eval_batch_size'])
        test_batch_size = eval(config['DATA']['eval_batch_size'])
        num_workers = eval(config['DATA']['num_workers'])
        gcn_layer_num = eval(config['MODEL'].get('gcn_layer_num', '1'))
        message_g = self.g
        # message_g, val_g, test_g, train_eids_dict = self._get_graphs_and_eids()

        if 'step_per_epoch' not in config['TRAIN'].keys():
            # 训练时进行采样
            # 分bin测试用
            if self.task == 'Rate':
                eid_for_loader = {('user', 'rate', 'item'): message_g.edges(etype='rate', form='eid')}
            elif self.task == 'Link':
                eid_for_loader = {('user', 'trust', 'user'): message_g.edges(etype='trust', form='eid')}
            else:   # joint
                eid_for_loader = {
                    ('user', 'rate', 'item'): message_g.edges(etype='rate', form='eid'),
                    ('user', 'trust', 'user'): message_g.edges(etype='trust', form='eid')
                }


            if self.bin_sep_lst is not None:
                self.get_bins_userid_lst(dataset)

            self._get_history_lst()  # 获取历史记录，构建负采样
            self.train_loader = self._get_loader(mode='train', g=message_g, eid_dict=eid_for_loader,
                                                 batch_size=train_batch_size, num_workers=num_workers, is_shuffle=True,
                                                 num_layers=gcn_layer_num, k=self.train_neg_num)
        else:
            # 训练时使用全图
            self.step_per_epoch = eval(config['TRAIN']['step_per_epoch'])
            # 分bin测试用
            if self.bin_sep_lst is not None:
                self.get_bins_userid_lst(dataset)

            self._get_history_lst()
            self.train_loader = range(eval(self.config['TRAIN']['step_per_epoch']))

        self.message_g = message_g.to(self.device)  # 用来测试时计算最终表示

        self.val_loader = self._get_loader(mode='evaluate', g=message_g,
                                           batch_size=val_batch_size, num_workers=num_workers, is_shuffle=False,
                                           num_layers=gcn_layer_num)

        self.test_loader = self._get_loader(mode='test', g=message_g,
                                            batch_size=test_batch_size, num_workers=num_workers, is_shuffle=False,
                                            num_layers=gcn_layer_num)

    def _get_model_specific_etype_graph(self, g):
        if self.task == 'Link':
            g = dgl.edge_type_subgraph(g, ['trust', 'trusted-by'])
        else:
            if self.model_name in cf_model:
                g = dgl.edge_type_subgraph(g, ['rate', 'rated-by'])
            elif self.model_name in social_model:
                g = dgl.edge_type_subgraph(g, ['rate', 'rated-by', 'trust', 'trusted-by'])
            elif self.model_name in directed_social_model:
                g = dgl.edge_type_subgraph(g, ['rate', 'rated-by', 'trusted-by'])
            else:
                raise ValueError("Wrong Model Name!!!")
        return g

    # def _get_graphs_and_eids(self):
    #     # assert self.task == 'Rate' or self.task == 'Link'
    #     # 如果是joint还需要改写，因为验证时rate和link各需要一个dataloader
    #     s_e_dict = {
    #         'train': {
    #             'rate': (0, self.train_rate_size),
    #             'link': (0, self.train_link_size),
    #         },
    #         'val': {
    #             'rate': (self.train_rate_size, self.train_rate_size + self.val_rate_size),
    #             'link': (self.train_link_size, self.train_link_size + self.val_link_size),
    #         },
    #         'test': {
    #             'rate': (self.train_rate_size + self.val_rate_size, -1),
    #             'link': (self.train_link_size + self.val_link_size, -1),
    #         }
    #     }
    #     eid_dict = {
    #         'train': {},
    #         'val': {},
    #         'test': {}
    #     }
    #
    #     if self.task == 'Rate':
    #         pred_etypes = [('user', 'rate', 'item')]
    #     elif self.task == 'Joint':
    #         pred_etypes = [('user', 'rate', 'item'), ('user', 'trust', 'user')]
    #     else:
    #         pred_etypes = [('user', 'trust', 'user')]
    #     # else:   # Joint
    #     #     pred_etypes = [('user', 'rate', 'item'), ('user', 'trust', 'user')]
    #
    #     for m in eid_dict.keys():
    #         # 训练时需要所有类型的边，验证时只需要rate或trust边
    #         if m == 'train':
    #             canonical_etypes = self.g.canonical_etypes
    #         else:
    #             canonical_etypes = pred_etypes
    #         for etype in canonical_etypes:
    #             if 'rate' in etype[1]:  # 1位置就是具体的etype名字
    #                 start, end = s_e_dict[m]['rate']
    #             else:
    #                 start, end = s_e_dict[m]['link']
    #             if end == -1:
    #                 end = self.g.num_edges(etype=etype)
    #             eid_dict[m].update({
    #                 etype: range(start, end),
    #             })
    #     message_g = dgl.edge_subgraph(self.g, eid_dict['train'], relabel_nodes=False)
    #     val_pred_g = dgl.edge_subgraph(self.g, eid_dict['val'], relabel_nodes=False)
    #     test_pred_g = dgl.edge_subgraph(self.g, eid_dict['test'], relabel_nodes=False)
    #
    #     eid_for_loader = {
    #         etype: message_g.edges(etype=etype[1], form='eid')
    #         for etype in pred_etypes
    #     }
    #     # if self.task == 'Rate':
    #     #     eid_for_loader = {('user', 'rate', 'item'): message_g.edges(etype='rate', form='eid')}
    #     # elif self.task == 'Link':
    #     #     eid_for_loader = {('user', 'trust', 'user'): message_g.edges(etype='trust', form='eid')}
    #     # else:
    #     #     eid_for_loader = {
    #     #         ('user', 'rate', 'item'): message_g.edges(etype='rate', form='eid'),
    #     #         ('user', 'trust', 'user'): message_g.edges(etype='trust', form='eid')
    #     #     }
    #     return message_g, val_pred_g, test_pred_g, eid_for_loader

    def _get_loader(self, mode, g, batch_size, num_workers, is_shuffle, **other_args):
        if self.device == torch.device('cuda'):
            g = g.to(self.device)
            num_workers = 0

        if mode == 'train':
            # 训练时进行邻居采样
            eid_dict = other_args['eid_dict']
            eid_dict = {k: v.to(self.device) for k, v in eid_dict.items()}
            num_layers = other_args['num_layers']
            assert 2 >= num_layers > 0
            if num_layers == 2:
                sample_nums = [10, 25]
            else:
                sample_nums = [10]
            k = other_args['k']
            sampler = dgl.dataloading.NeighborSampler(sample_nums)
            # 因为这里的g是只有message g和验证边的
            # 所以这里需要排除的边就是需要验证的边
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                sampler, negative_sampler=NegativeSampler(self.history_lst, self.total_num, k),
            )
            dataloader = dgl.dataloading.DataLoader(
                g, eid_dict, sampler,
                batch_size=batch_size,
                shuffle=is_shuffle,
                drop_last=False,
                num_workers=num_workers)

        elif mode == 'evaluate' or mode == 'test':
            # 测试时，在全体负样本上进行召回
            # 那么，需要迭代所有的userID，然后根据id返回多层邻居子图，不需要负采样
            # 这里应该比较像节点预测的返回情况，不需要转成链接预测
            # 要注意，此时测试输入的图仍然是训练图
            num_layers = other_args['num_layers']
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
            dataloader = dgl.dataloading.DataLoader(
                g, {'user': g.nodes(ntype='user')}, sampler,
                batch_size=batch_size,
                shuffle=is_shuffle,
                drop_last=False,
                num_workers=num_workers)
        else:
            raise ValueError("Wrong Mode!!")
        return dataloader

    def _get_history_lst(self):
        # 训练负采样用
        with open(f'./data/{self.data_name}/splited_data/user2v.json', 'r', encoding='utf-8') as f:
            user2v = json.load(f)
        with open(f'./data/{self.data_name}/splited_data/gt.json', 'r', encoding='utf-8') as f:
            gt = json.load(f)

        user2item = user2v['user2item']
        user2trust = user2v['user2trust']
        self.u_lst = nList()
        item_lst = nList()
        trust_lst = nList()
        self.history_lst = {}
        self.total_num = {}
        assert user2trust.keys() == user2item.keys()

        if self.task == 'Rate' or self.task == 'Joint':
            for u, v in user2item.items():
                self.u_lst.append(int(u))
                if len(v) == 0:
                    v = [-1024]
                item_lst.append(nList(v))
            self.history_lst.update({
                'rate': item_lst
            })
            self.total_num.update({
                'rate': self.item_num
            })
        if self.task == 'Link' or self.task == 'Joint':
            for u, v in user2trust.items():
                if self.task == 'Link':
                    self.u_lst.append(int(u))
                v.append(int(u))
                trust_lst.append(nList(v))
            self.history_lst.update({
                'trust': trust_lst
            })
            self.total_num.update({
                'trust': self.user_num
            })

        # 测试用
        self.pos_items = {
            k: np.array([list(l) for l in v], dtype=object)
            for k, v in self.history_lst.items()
        }
        if self.task == 'Rate':
            self.gt = {
                'rate': {
                    'evaluate': np.array(gt['rate']['val'], dtype=object),
                    'test': np.array(gt['rate']['test'], dtype=object)
                }
            }
        elif self.task == 'Link':
            self.gt = {
                'trust': {
                    'evaluate': np.array(gt['link']['val'], dtype=object),
                    'test': np.array(gt['link']['test'], dtype=object)
                }
            }
        else:
            self.gt = {
                'rate': {
                    'evaluate': np.array(gt['rate']['val'], dtype=object),
                    'test': np.array(gt['rate']['test'], dtype=object)
                },
                'trust': {
                    'evaluate': np.array(gt['link']['val'], dtype=object),
                    'test': np.array(gt['link']['test'], dtype=object)
                }
            }


    def _print_config(self):
        # 用来打印config信息
        config_str = ''
        config_str += '=' * 10 + "Config" + '=' * 10 + '\n'
        for k, v in self.config.items():
            if k == 'VISUALIZED' or k == 'LOG':
                continue
            config_str += k + ': \n'
            for _k, _v in v.items():
                config_str += f'\t{_k}: {_v}\n'
        config_str += ('=' * 25 + '\n')
        self.print_info(self._log(config_str, mode='w'))

    def _to(self, device=None):
        # 整体迁移
        if device is None:
            self.model = self.model.to(self.config['TRAIN']['device'])
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
            self.config['TRAIN']['device'] = device

    def print_best_metrics(self):
        metric_str = ''
        for t in self.metric.task_lst:
            for m in self.metric.metric_name:
                for k in self.metric.ks:
                    v = self.metric.metric_dict[t][m][k]['best']
                    metric_str += f'{t} best: {m}@{k}: {v:.4f}\t'

                    # if self.metric.bin_sep_lst is not None:
                    #     metric_str += '\n'
                    #     # bins_best_metric_value = self.metric.metric_dict[t][m][k]['bin_best']
                    #     # info_table = self.generate_bins_table_info(
                    #     #     bins_best_metric_value,
                    #     #     f'{m}@{k}'
                    #     # )
                    #     # metric_str += info_table
                    #     for i in range(len(self.metric.bin_sep_lst)):
                    #         v = self.metric.metric_dict[t][m][k]['bin_best'][i]
                    #         s = self.metric.bin_sep_lst[i]
                    #         if i == len(self.metric.bin_sep_lst) - 1:
                    #             e = '∞'
                    #         else:
                    #             e = self.metric.bin_sep_lst[i + 1]
                    #         metric_str += f'{t} best: {m}@{k} in {s}-{e}: {v:.4f}\t'
                    #     metric_str += '\n'
                metric_str += '\n'
        return metric_str

    def _generate_metric_str(self, metric_str, is_val=True, print_best=False):
        # 根据metric结果，生成文本
        for t in self.metric.metric_dict.keys():
            metric_table = [[t] + self.ks]
            for m in self.metric.metric_dict[t].keys():
                metric_vis_dict = {}
                bin_metric_vis_dict = {}
                cur_row = [m]
                for k in self.metric.metric_dict[t][m].keys():
                    if print_best:
                        v = self.metric.metric_dict[t][m][k]['best']
                    else:
                        v = self.metric.metric_dict[t][m][k]['value']

                    cur_row.append(v)
                    if self.is_visulized:
                        metric_vis_dict[f'{m}@{k}'] = v
                    # metric_str += f'{t} {m}@{k}: {v:.4f}\t'


                    if self.bin_sep_lst is not None and not is_val:
                        metric_str += '\n'
                        bins_test_metric_value = self.metric.metric_dict[t][m][k]['bin_value']
                        info_table = self.generate_bins_table_info(
                            bins_test_metric_value,
                            f'{m}@{k}'
                        )
                        metric_str += info_table + '\n'
                        # bin_metric_vis_dict[k] = {}
                        # for i in range(len(self.bin_sep_lst)):
                        #     v = self.metric.metric_dict[t][m][k]['bin_value'][i]
                        #     s = self.bin_sep_lst[i]
                        #     if i == len(self.bin_sep_lst) - 1:
                        #         e = '∞'
                        #     else:
                        #         e = self.bin_sep_lst[i + 1]
                        #
                        #     if self.is_visulized:
                        #         bin_metric_vis_dict[k][f'{m}@{k}_in_{s}-{e}'] = v
                        #
                        #     metric_str += f'{t} {m}@{k} in {s}-{e}: {v:.4f}\t'
                        # metric_str += '\n'
                metric_table.append(cur_row)
                if self.is_visulized and is_val:
                    self.writer.add_scalars(f'{m}/total_{m}', metric_vis_dict, self.vis_cnt)
                    # if self.bin_sep_lst is not None:
                    #     for k in bin_metric_vis_dict.keys():
                    #         self.writer.add_scalars(f'{m}/bin_{m}/{m}@{k}', bin_metric_vis_dict[k], self.vis_cnt)
                # metric_str += '\n'
            metric_str += tabulate(metric_table, headers='firstrow', tablefmt='simple', floatfmt='.4f')
            metric_str += '\n\n'
        if self.is_visulized:
            self.vis_cnt += 1
        # if is_val:
        #     self.metric.clear_metrics()
        return metric_str

    # def _generate_metric_str(self, metric_str, is_val=True):
    #     # 根据metric结果，生成文本
    #     for t in self.metric.metric_dict.keys():
    #         for m in self.metric.metric_dict[t].keys():
    #             metric_vis_dict = {}
    #             bin_metric_vis_dict = {}
    #             for k in self.metric.metric_dict[t][m].keys():
    #                 v = self.metric.metric_dict[t][m][k]['value']
    #                 if self.is_visulized:
    #                     metric_vis_dict[f'{m}@{k}'] = v
    #                 metric_str += f'{t} {m}@{k}: {v:.4f}\t'
    #
    #
    #                 if self.bin_sep_lst is not None and not is_val:
    #                     metric_str += '\n'
    #                     bins_test_metric_value = self.metric.metric_dict[t][m][k]['bin_value']
    #                     info_table = self.generate_bins_table_info(
    #                         bins_test_metric_value,
    #                         f'{m}@{k}'
    #                     )
    #                     metric_str += info_table
    #                     # bin_metric_vis_dict[k] = {}
    #                     # for i in range(len(self.bin_sep_lst)):
    #                     #     v = self.metric.metric_dict[t][m][k]['bin_value'][i]
    #                     #     s = self.bin_sep_lst[i]
    #                     #     if i == len(self.bin_sep_lst) - 1:
    #                     #         e = '∞'
    #                     #     else:
    #                     #         e = self.bin_sep_lst[i + 1]
    #                     #
    #                     #     if self.is_visulized:
    #                     #         bin_metric_vis_dict[k][f'{m}@{k}_in_{s}-{e}'] = v
    #                     #
    #                     #     metric_str += f'{t} {m}@{k} in {s}-{e}: {v:.4f}\t'
    #                     metric_str += '\n'
    #
    #             if self.is_visulized and is_val:
    #                 self.writer.add_scalars(f'{m}/total_{m}', metric_vis_dict, self.vis_cnt)
    #                 if self.bin_sep_lst is not None:
    #                     for k in bin_metric_vis_dict.keys():
    #                         self.writer.add_scalars(f'{m}/bin_{m}/{m}@{k}', bin_metric_vis_dict[k], self.vis_cnt)
    #             metric_str += '\n'
    #
    #     if self.is_visulized:
    #         self.vis_cnt += 1
    #     # if is_val:
    #     #     self.metric.clear_metrics()
    #     return metric_str

    def _log(self, str_, mode='a'):
        # 将log写入文件
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def _save_model(self, save_pth):
        # 保存最好模型
        if self.task == 'Joint':
            task = 'Rate'
        else:
            task = self.task

        best_value = self.metric.metric_dict[task][self.save_metric][self.indicator_k]['best']
        self.print_info(self._log(f"Save Best {self.save_metric}@{self.indicator_k} {best_value:.4f} Model!"))
        dir_pth, _ = os.path.split(save_pth)
        if not os.path.isdir(dir_pth):
            father_dir_pth, _ = os.path.split(dir_pth)
            if not os.path.isdir(father_dir_pth):
                os.mkdir(father_dir_pth)
            os.mkdir(dir_pth)
        torch.save(self.model.state_dict(), save_pth)

    def _load_model(self, save_pth, strict=False):
        if self.task == 'Joint':
            task = 'Rate'
        else:
            task = self.task

        best_value = self.metric.metric_dict[task][self.save_metric][self.indicator_k]['best']
        self.print_info(self._log(f"Load Best {self.save_metric} {best_value:.4f} Model!"))
        state_dict = torch.load(save_pth)
        self.model.load_state_dict(state_dict, strict=strict)


    # def get_bins_userid_lst(self, message_g):
    #     if self.task == 'Rate':
    #         etype = 'rate'
    #     else:
    #         etype = 'trust'
    #
    #     # 根据message g确定用户所属的类别
    #     # 分组内保存的是用户的id
    #
    #     # 分组起点终点
    #     self.bins_start_end_lst = []
    #     # u_num_lst = []
    #     self.u_lst = []
    #     for i in range(len(self.bin_sep_lst)):
    #         # 获取对应的mask，找到train_g里面在当前度范围内的用户id
    #         s = self.bin_sep_lst[i]
    #         if i == len(self.bin_sep_lst) - 1:
    #             mask = (message_g.out_degrees(etype=etype) >= s)
    #             e = '∞'
    #         else:
    #             e = self.bin_sep_lst[i + 1]
    #             mask = (message_g.out_degrees(etype=etype) >= s) * \
    #                    (message_g.out_degrees(etype=etype) < e)
    #         self.bins_start_end_lst.append((s, e))
    #         # 所有在这个度范围内的用户id tensor
    #         u_lst = message_g.nodes('user').masked_select(mask)
    #         self.print_info(self._log(f'{s}-{e}: {len(u_lst)}'))
    #         self.u_lst.append(u_lst)
    #     self.metric.bins_id_lst = self.u_lst
    #     self.bins_id_lst = self.u_lst

    def generate_bins_table_info(self, info, title):
        info_table = []
        for i in range(len(self.bin_sep_lst) + 2):
            info_table.append([])
            cur_row = info_table[-1]
            for j in range(len(self.bin_sep_lst) + 2):
                if i == 0:
                    if j == 0:
                        cur_row.append(title)
                    elif j == len(self.bin_sep_lst) + 1:
                        cur_row.append('Row Sum')
                    else:
                        # 空位是rate_s和rate_e
                        _, _, link_s, link_e = self.bins_start_end_lst[0][j - 1]
                        cur_row.append(f'L: {link_s}-{link_e}')
                else:
                    if j == 0:
                        if i == len(self.bin_sep_lst) + 1:
                            cur_row.append('Col Sum')
                        else:
                            rate_s, rate_e, _, _ = self.bins_start_end_lst[i - 1][0]
                            cur_row.append(f'R: {rate_s}-{rate_e}')
                    else:
                        if isinstance(info[i - 1][j - 1], list):
                            cur_row.append(f'{"/".join([str(n) for n in info[i - 1][j - 1]])}')
                        else:
                            cur_row.append(f'{info[i - 1][j - 1]}')
        return tabulate(info_table, headers='firstrow', tablefmt='fancy_grid', floatfmt='.4f')

    def get_bins_userid_lst(self, dataset):
        self.bins_start_end_lst = dataset.bins_start_end_lst
        self.bins_id_lst = dataset.u_lst
        self.metric.bins_id_lst = dataset.u_lst
        self.bin_user_num = dataset.bin_user_num
        self.rate_edges_num = dataset.rate_edges_num
        self.link_edges_num = dataset.link_edges_num
        bins_user_edges_num = []
        for i in range(len(self.bin_sep_lst) + 1):  #+1是多出来的汇总行/列
            bins_user_edges_num.append([])
            for j in range(len(self.bin_sep_lst) + 1):
                bins_user_edges_num[-1].append(
                    [self.bin_user_num[i][j], self.rate_edges_num[i][j], self.link_edges_num[i][j]]
                )

        # bins_user_num = \
        #     [[len(self.bins_id_lst[i][j]) for j in range(len(self.bin_sep_lst))] for i in range(len(self.bin_sep_lst))]
        info_table = self.generate_bins_table_info(
            bins_user_edges_num,
            '#User/#Rate/#Link'
        )
        self.print_info(self._log('User Rate Degree and Link Degree Distribution Table'))
        self.print_info(self._log(info_table))

    def get_full_ratings(self, batch_users, message_g=None, input_nodes=None, task=None, mode='val'):
        # batch_users: 确保有序，因为这样获取pos item和gt的时候，就可以直接切片操作
        # 在全样本空间上计算概率，参考SocialLGN的代码
        ## 获取batch users的起始的终止id
        min_user = torch.min(batch_users)
        max_user = torch.max(batch_users)
        if task == 'Rate':
            etype = 'rate'
        else:
            etype = 'trust'

        all_pos_items = self.pos_items[etype][min_user: max_user + 1] # 调用函数，输入batchusers，返回一个包含所有互动过物品的id列表
        gt = self.gt[etype][mode][min_user: max_user + 1] # 调用函数，输出batch users，返回测试集中的互动列表，这个是开区间
        assert torch.tensor([len(gt[i]) for i in range(len(gt))]).min() > 0
        batch_users = batch_users.to(self.device)
        # 计算batch user在所有item上的评分
        ## 获取embeddings
        ### user_embedding + item_embedding/user_embedding
        u_embed, v_embed, u_bias, v_bias= self.model.get_final_embeddings(batch_users, etype)
        ### all rating
        rating = torch.matmul(u_embed, v_embed.t())
        assert rating.shape == (len(batch_users), v_embed.shape[0])

        # SVD用
        if u_bias is not None:
            assert u_bias.shape == (len(batch_users), 1)
            assert v_bias.shape == (v_embed.shape[0], 1)
            rating = rating + u_bias
            del u_bias
            rating = rating + v_bias.t()
            del v_bias

        # 计算负样本列表
        ## 这里是建一个一个id,item的列表，从而将rating矩阵中(id,item)位置的rating mask
        exclude_idx = []
        exclude_items = []
        for id, items in enumerate(all_pos_items):
            if items == [-1024]:
                continue
            exclude_idx.extend([id] * len(items))
            exclude_items.extend(items)
        rating[exclude_idx, exclude_items] = -(1 << 10)
        _, rating_k = torch.topk(rating, k=max(self.ks))
        del rating
        # 这里需要一个rating lst，一个gt lst
        return batch_users, rating_k.cpu(), gt

    def get_side_info(self):
        return None

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if self.task == 'Rate':
            etype = ('user', 'rate', 'item')
        elif self.task == 'Link':
            etype = ('user', 'trust', 'user')
        else:
            etype = [('user', 'rate', 'item'), ('user', 'trust', 'user')]

        if mode == 'train':
            # 采样
            input_nodes, pos_pred_g, neg_pred_g, blocks = inputs['graphs']
            if isinstance(input_nodes, dict):
                input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
            else:
                input_nodes = input_nodes.to(self.device)
            message_g = [b.to(self.device) for b in blocks]
            pos_pred_g = pos_pred_g.to(self.device)
            neg_pred_g = neg_pred_g.to(self.device)
            cur_step = inputs['cur_step']

            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(
                message_g,
                pos_pred_g,
                neg_pred_g,
                input_nodes=input_nodes
            )
            if len(output) == 2:
                pos_pred, neg_pred = output
                neg_pred = neg_pred.reshape(-1, self.train_neg_num)
                loss = self.loss_func(pos_pred, neg_pred)
                loss.backward()
                self.optimizer.step()
                return loss.item()
            else:
                pos_rate_pred, neg_rate_pred, pos_link_pred, neg_link_pred = output
                neg_rate_pred = neg_rate_pred.reshape(-1, self.train_neg_num)
                neg_link_pred = neg_link_pred.reshape(-1, self.train_neg_num)
                rate_loss = self.loss_func(pos_rate_pred, neg_rate_pred)
                link_loss = self.loss_func(pos_link_pred, neg_link_pred)
                loss = rate_loss + link_loss
                loss.backward()
                self.optimizer.step()
                return loss.item(), rate_loss.item(), link_loss.item()

        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                input_nodes, output_nodes, blocks = inputs['graphs']
                if isinstance(input_nodes, dict):
                    input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
                else:
                    input_nodes = input_nodes.to(self.device)

                output_nodes = {k: v.to(self.device) for k, v in output_nodes.items()}
                message_g = [b.to(self.device) for b in blocks]

                self.model.eval()
                if self.task != 'Joint':
                    pred_task_lst = [self.task]
                else:
                    pred_task_lst = ['Rate', 'Link']
                for task in pred_task_lst:
                    batch_users, rating_k, gt = \
                        self.get_full_ratings(
                            batch_users=output_nodes['user'],
                            message_g=message_g,
                            input_nodes=input_nodes,
                            task=task,
                            mode=mode
                        )
                    is_test = True if mode == 'test' else False
                    self.metric.compute_metrics(batch_users.cpu(), rating_k, gt, task=task, is_test=is_test)
                if self.task == 'Joint':
                    return torch.nan, torch.nan, torch.nan
                else:
                    return torch.nan
        else:
            raise ValueError("Wrong Mode")

    def wrap_step_loop(self, mode, data_loader: dgl.dataloading.DataLoader or range, side_info, loss_name):
        all_loss_lst = [0.0 for _ in range(len(loss_name))]

        val_true_batch_cnt = [0 for _ in range(len(loss_name))]
        if self.is_log and mode == 'train':
            bar_loader = tqdm(enumerate(data_loader), total=len(data_loader))
        else:
            bar_loader = enumerate(data_loader)
        if mode != 'train':
            self.metric.iter_step = 0
            # 计算所有user item的最终表示
            self.model.compute_final_embeddings(self.message_g)

        for i, graphs in bar_loader:
            loss_lst = self.step(
                mode=mode, graphs=graphs,
                side_info=side_info, cur_step=i)
            for j in range(len(loss_name)):
                if len(loss_name) == 1:
                    loss_lst = [loss_lst]
                # 因为如果需要测试两种边，dataloader会依次读入两种类型，如果此时将没有出现的loss记为0，会导致数据不准
                # 做法是，rate批次设置link loss为0，link批次设置rate loss为0
                # 因此，需要记录rate批次和link批次的次数，一个epoch完成后再进行校正
                if loss_lst[j] != -1:
                    val_true_batch_cnt[j] += 1
                    all_loss_lst[j] += loss_lst[j]
                else:
                    all_loss_lst[j] += 0
            if mode != 'train':
                self.metric.iter_step += 1

        for j in range(len(loss_name)):
            # if val_true_batch_cnt[j] != 0:
            #     # 说明loss需要校正
            #     all_loss_lst[j] /= len(data_loader)
            # all_loss_lst[j] /= len(data_loader)

            # 如果出现rate和link的数量和不等于total的，不用担心，因为会有一个batch包含两类边
            all_loss_lst[j] /= val_true_batch_cnt[j]
        return all_loss_lst

    def train(self):
        if self.task == 'Joint':
            self.loss_name = ['Total Loss', 'Rate Loss', 'Link Loss']
        else:
            self.loss_name = ['Loss']
        return self._train(self.loss_name)

    def _train(self, loss_name, side_info: dict=None):
        # 整体训练流程
        self.print_info(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])

        side_info = self.get_side_info()
        for e in range(1, epoch + 1):
            """
            write codes for train
            and return loss
            """
            self.cur_e = e

            # train
            all_loss_lst = self.wrap_step_loop('train', self.train_loader, side_info, loss_name)
            metric_str = f'\nEpoch: {e}\n'
            for j in range(len(loss_name)):
                metric_str += f'{loss_name[j]}: {all_loss_lst[j]:.4f}\t'
            self.print_info(self._log(metric_str))

            if e % self.eval_step == 0:
                # 在训练图上跑节点表示，在验证图上预测边的概率
                self.metric.clear_metrics()
                all_loss_lst = self.wrap_step_loop('evaluate', self.val_loader, side_info, loss_name)
                self.metric.get_batch_metrics()
                # metric_str = f'Evaluate Epoch: {e}\n'
                # for j in range(len(loss_name)):
                #     metric_str += f'{loss_name[j]}: {all_loss_lst[j]:.4f}\t'
                # metric_str += '\n'
                metric_str = self._generate_metric_str('')
                self.print_info(self._log(metric_str))

                # 保存最好模型
                if self.metric.is_save:
                    self._save_model(self.save_pth)
                    self.metric.is_save = False

                if self.trial is not None:
                    intermediate_value = self.metric.metric_dict[self.indicator_task][self.save_metric][self.indicator_k]['value']
                    self.trial.report(intermediate_value, e)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()

                # 是否早停
                indicator_metric = self.metric.metric_dict[self.indicator_task][self.save_metric][self.indicator_k][
                    'value']
                self.early_stopper(indicator_metric)
                if e > self.warm_epoch:
                    if self.early_stopper.early_stop:
                        self.print_info(self._log("Early Stop!"))
                        break
                else:
                    self.early_stopper.counter = 0

                if self.early_stopper.counter > 0:
                    self.print_info(
                        self._log(f'EarlyStopping counter: {self.early_stopper.counter} out of {self.early_stopper.patience}'))
                # else:
                #     self.metric.is_early_stop = False

                if self.is_visulized:
                    for j in range(len(loss_name)):
                        self.writer.add_scalar(f'loss/Train_{loss_name[j]}', all_loss_lst[j], e)
                        self.writer.add_scalar(f'loss/Val_{loss_name[j]}', val_loss_lst[j], e)

            # self.lr_scheduler.step()
        self.print_info(self._log(self._generate_metric_str('\nBest Val Metric\n', print_best=True)))
        # self.print_info(self._log(self.print_best_metrics()))

        if self.task == 'Joint':
            task = 'Rate'
        else:
            task = self.task
        val_best_nDCG = self.metric.metric_dict[task][self.save_metric][self.indicator_k]['best']

        if self.trial is None:
            # 搜索参数时不能用测试集
            # 开始测试
            # 加载最优模型
            self._load_model(self.save_pth)
            self.metric.clear_metrics()

            all_loss_lst = self.wrap_step_loop('test', self.test_loader, side_info, loss_name)
            self.metric.get_batch_metrics(is_test=True)
            metric_str = f'\nTest Epoch: \n'
            # for j in range(len(loss_name)):
            #     metric_str += f'{loss_name[j]}: {all_loss_lst[j]:.4f}\t'
            # metric_str += '\n'
            metric_str = self._generate_metric_str(metric_str, is_val=False)
            self.print_info(self._log(metric_str))
        self.print_info("=" * 10 + "TRAIN END" + "=" * 10)
        return val_best_nDCG