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

from tqdm import tqdm, trange

import time
import os
import json
import copy

cf_model = ['LightGCN', 'MF']
social_model = ['MutualRec', 'FusionLightGCN', 'DiffnetPP', 'GraphRec']
directed_social_model = ['TrustSVD', 'SVDPP', 'Sorec', 'SocialMF']
link_model = ['AA', 'Node2Vec']



"""
BaseTrainer主要用来写一些通用的函数，比如打印config之类
"""

class BaseTrainer:
    def __init__(self, config):
        self.task = config['TRAIN']['task']
        self.is_visulized = config['VISUALIZED']
        self.is_log = config['LOG']

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

        self.bin_sep_lst = eval(config['METRIC'].get('bin_sep_lst', 'None'))


        if self.is_visulized:
            tensor_board_dir = os.path.join(
                log_dir, self.model_name, self.data_name, f'{self.task}_{self.random_seed}_{self.model_name}')
            self.writer = SummaryWriter(tensor_board_dir)
            tqdm.write(self._log(f'tensor_board_pth: {tensor_board_dir}'))
            self.vis_cnt = 1

        self.model_name = config['MODEL']['model_name']
        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.neg_num = eval(config['DATA']['neg_num'])
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

        lr_scheduler_name = 'torch.optim.lr_scheduler.' + config['OPTIM']['lr_scheduler']
        T_0 = eval(config['OPTIM']['T_0'])  # 学习率第一次重启的epoch数
        T_mult = eval(config['OPTIM']['T_mult'])  # 学习率衰减epoch数变化倍率
        self.lr_scheduler = eval(lr_scheduler_name)(self.optimizer, T_0=T_0, T_mult=T_mult, verbose=True)

    def _build_metric(self):
        # metric
        self.metric = BaseMetric(self.config)

    def _build_data(self):
        config = self.config
        # get data loader
        dataset = DGLRecDataset(config)
        self.etypes = dataset[0].etypes
        self.g = dataset[0]
        self.g = self._get_model_specific_etype_graph(self.g)

        self.train_rate_size = dataset.train_rate_size
        self.val_rate_size = dataset.val_rate_size
        self.train_link_size = dataset.train_link_size
        self.val_link_size = dataset.val_link_size

        train_batch_size = eval(config['DATA']['train_batch_size'])
        val_batch_size = eval(config['DATA']['eval_batch_size'])
        test_batch_size = eval(config['DATA']['eval_batch_size'])
        num_workers = eval(config['DATA']['num_workers'])
        budget = eval(config['DATA'].get('budget', '1000'))
        gcn_layer_num = eval(config['MODEL'].get('gcn_layer_num', '1'))

        message_g, val_g, test_g, eids_dict = self._get_graphs_and_eids()

        self._get_history_lst()
        # eids_dict = self._get_eids()
        self.train_loader = self._get_loader(mode='train', g=message_g, eid_dict=eids_dict['train'],
                                             batch_size=train_batch_size, num_workers=num_workers, is_shuffle=True,
                                             budget=budget)

        self.val_loader = self._get_loader(mode='evaluate', g=val_g, eid_dict=eids_dict['val'],
                                           batch_size=val_batch_size, num_workers=0, is_shuffle=False,
                                           num_layers=gcn_layer_num, k=self.neg_num)

        self.test_loader = self._get_loader(mode='test', g=test_g, eid_dict=eids_dict['test'],
                                            batch_size=test_batch_size, num_workers=0, is_shuffle=False,
                                            num_layers=gcn_layer_num, k=self.neg_num)

        # 分bin测试用
        if self.bin_sep_lst is not None:
            self.train_e_id_lst, self.val_e_id_lst, self.test_e_id_lst = self.get_bins_eid_lst(
                message_g, val_g, test_g, eids_dict)

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

    def _get_graphs_and_eids(self):
        assert self.task == 'Rate' or self.task == 'Link'
        # 如果是joint还需要改写，因为验证时rate和link各需要一个dataloader
        s_e_dict = {
            'train': {
                'rate': (0, self.train_rate_size),
                'link': (0, self.train_link_size),
            },
            'val': {
                'rate': (self.train_rate_size, self.train_rate_size + self.val_rate_size),
                'link': (self.train_link_size, self.train_link_size + self.val_link_size),
            },
            'test': {
                'rate': (self.train_rate_size + self.val_rate_size, -1),
                'link': (self.train_link_size + self.val_link_size, -1),
            }
        }
        eid_dict = {
            'train': {},
            'val': {},
            'test': {}
        }

        if self.task == 'Rate':
            pred_etypes = [('user', 'rate', 'item')]
        else:
            pred_etypes = [('user', 'trust', 'user')]
        # else:   # Joint
        #     pred_etypes = [('user', 'rate', 'item'), ('user', 'trust', 'user')]

        for m in eid_dict.keys():
            # 训练时需要所有类型的边，验证时只需要rate或trust边
            if m == 'train':
                canonical_etypes = self.g.canonical_etypes
            else:
                canonical_etypes = pred_etypes
            for etype in canonical_etypes:
                if 'rate' in etype[1]:  # 1位置就是具体的etype名字
                    start, end = s_e_dict[m]['rate']
                else:
                    start, end = s_e_dict[m]['link']
                if end == -1:
                    end = self.g.num_edges(etype=etype)
                eid_dict[m].update({
                    etype: range(start, end),
                })
        message_g = dgl.edge_subgraph(self.g, eid_dict['train'], relabel_nodes=False)
        val_pred_g = dgl.edge_subgraph(self.g, eid_dict['val'], relabel_nodes=False)
        test_pred_g = dgl.edge_subgraph(self.g, eid_dict['test'], relabel_nodes=False)
        # 包含了消息边的测试图
        e = pred_etypes[0]
        # 这里，pred边被加到message边后面
        val_g = message_g.clone()
        val_g.add_edges(*val_pred_g.edges(etype=e), etype=e)
        test_g = message_g.clone()
        test_g.add_edges(*test_pred_g.edges(etype=e), etype=e)

        train_size = len(eid_dict['train'][pred_etypes[0]])
        val_size = len(eid_dict['val'][pred_etypes[0]])
        test_size = len(eid_dict['test'][pred_etypes[0]])
        eid_dict_for_dataloader = {
            'train': {e: range(train_size)},  # 只选择一种边计数，避免对比时joint模型训练时间更长
            'val': {e: range(train_size, train_size + val_size)},
            'test': {e: range(train_size, train_size + test_size)},
        }
        return message_g, val_g, test_g, eid_dict_for_dataloader

    def _get_loader(self, mode, g, eid_dict, batch_size, num_workers, is_shuffle, **other_args):
        if self.device == torch.device('cuda'):
            g = g.to(self.device)
            eid_dict = {k: torch.tensor(v, device=self.device) for k, v in eid_dict.items()}
            num_workers = 0

        if mode == 'train':
            budget = other_args['budget']
            # 训练时使用graphsaint采样子图
            sampler = SAINTSamplerForHetero(mode='node', budget=budget)
            dataloader = dgl.dataloading.DataLoader(
                g, eid_dict, sampler, num_workers=num_workers, shuffle=is_shuffle, batch_size=batch_size, device=self.device
            )
        elif mode == 'evaluate' or mode == 'test':
            # 测试时使用full neighbour采样全图
            num_layers = other_args['num_layers']
            k = other_args['k']
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
            # 因为这里的g是只有message g和验证边的
            # 所以这里需要排除的边就是需要验证的边
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                sampler, negative_sampler=NegativeSampler(self.history_lst, self.total_num, k),
                exclude=lambda x: eid_dict
            )
            dataloader = dgl.dataloading.DataLoader(
                g, eid_dict, sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                device = self.device,
                num_workers=num_workers)
        else:
            raise ValueError("Wrong Mode!!")
        return dataloader

    def _get_eids(self):
        # joint 需要修改
        eids_dict = {}
        if self.task == 'Rate' or self.task == 'Joint':
            eids_dict = {
                'train': {('user', 'rate', 'item'): range(self.train_rate_size)},
                'val': {('user', 'rate', 'item'): range(self.train_rate_size, self.train_rate_size + self.val_rate_size)},
                'test': {('user', 'rate', 'item'): range(self.train_rate_size + self.val_rate_size, self.g.num_edges(('user', 'rate', 'item')))},
            }
        if self.task == 'Link' or self.task == 'Joint':
            eids_dict.update({
                'train': {('user', 'trust', 'user'): range(self.train_link_size)},
                'val': {('user', 'trust', 'user'): range(self.train_link_size, self.train_link_size + self.val_link_size)},
                'test': {('user', 'trust', 'user'): range(self.train_link_size + self.val_link_size, self.g.num_edges(('user', 'trust', 'user')))},
            })
        return eids_dict

    def _get_history_lst(self):
        # 获取用户历史行为列表
        with open(f'./data/{self.data_name}/splited_data/user2v.json', 'r', encoding='utf-8') as f:
            user2v = json.load(f)

        user2item = user2v['user2item']
        user2trust = user2v['user2trust']
        u_lst = nList()
        item_lst = nList()
        trust_lst = nList()
        self.history_lst = {}
        self.total_num = {}
        assert user2trust.keys() == user2item.keys()

        if self.task == 'Rate' or self.task == 'Joint':
            for u, v in user2item.items():
                u_lst.append(int(u))
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
                    u_lst.append(int(u))
                v.append(int(u))
                trust_lst.append(nList(v))
            self.history_lst.update({
                'trust': trust_lst
            })
            self.total_num.update({
                'trust': self.user_num
            })

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

    def _generate_metric_str(self, metric_str, is_val=True):
        # 根据metric结果，生成文本
        for t in self.metric.metric_dict.keys():
            for m in self.metric.metric_dict[t].keys():
                metric_vis_dict = {}
                bin_metric_vis_dict = {}
                for k in self.metric.metric_dict[t][m].keys():
                    v = self.metric.metric_dict[t][m][k]['value']
                    if self.is_visulized:
                        metric_vis_dict[f'{m}@{k}'] = v
                    metric_str += f'{t} {m}@{k}: {v:.4f}\t'

                    if self.bin_sep_lst is not None:
                        metric_str += '\n'
                        bin_metric_vis_dict[k] = {}
                        for i in range(len(self.bin_sep_lst)):
                            v = self.metric.metric_dict[t][m][k]['bin_value'][i]
                            s = self.bin_sep_lst[i]
                            if i == len(self.bin_sep_lst) - 1:
                                e = '∞'
                            else:
                                e = self.bin_sep_lst[i + 1]

                            if self.is_visulized:
                                bin_metric_vis_dict[k][f'{m}@{k}_in_{s}-{e}'] = v

                            metric_str += f'{t} {m}@{k} in {s}-{e}: {v:.4f}\t'
                        metric_str += '\n'

                if self.is_visulized and is_val:
                    self.writer.add_scalars(f'{m}/total_{m}', metric_vis_dict, self.vis_cnt)
                    if self.bin_sep_lst is not None:
                        for k in bin_metric_vis_dict.keys():
                            self.writer.add_scalars(f'{m}/bin_{m}/{m}@{k}', bin_metric_vis_dict[k], self.vis_cnt)
                metric_str += '\n'

        if self.is_visulized:
            self.vis_cnt += 1
        # if is_val:
        #     self.metric.clear_metrics()
        return metric_str

    def _log(self, str_, mode='a'):
        # 将log写入文件
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def _save_model(self, save_pth):
        # 保存最好模型
        best_value = self.metric.metric_dict[self.task][self.metric.save_metric][10]['best']
        tqdm.write(self._log(f"Save Best {self.metric.save_metric} {best_value:.4f} Model!"))
        dir_pth, _ = os.path.split(save_pth)
        if not os.path.isdir(dir_pth):
            father_dir_pth, _ = os.path.split(dir_pth)
            if not os.path.isdir(father_dir_pth):
                os.mkdir(father_dir_pth)
            os.mkdir(dir_pth)
        torch.save(self.model.state_dict(), save_pth)

    def _load_model(self, save_pth, strict=False):
        best_value = self.metric.metric_dict[self.task][self.metric.save_metric][10]['best']
        tqdm.write(self._log(f"Load Best {self.metric.save_metric} {best_value:.4f} Model!"))
        state_dict = torch.load(save_pth)
        self.model.load_state_dict(state_dict, strict=strict)

    def log_pred_histgram(self, pos_pred, neg_pred, mode='train'):
        pos_pred, neg_pred = pos_pred.detach().cpu(), torch.mean(neg_pred.detach(), dim=1, keepdim=True).cpu()
        cat_pred = torch.cat([pos_pred, neg_pred], dim=1)
        global_step = self.cur_e if mode != 'test' else 1
        self.writer.add_histogram(f'{mode}/total_pos_pred', pos_pred, global_step=global_step)
        self.writer.add_histogram(f'{mode}/total_neg_pred', neg_pred, global_step=global_step)
        self.writer.add_histogram(f'{mode}/total_cat_pred', cat_pred, global_step=global_step)
        if self.bin_sep_lst is not None:
            ## bin histgram
            for i in range(len(self.bins_id_lst)):
                id_lst = self.bins_id_lst[i]
                s, e = self.bins_start_end_lst[i]
                self.writer.add_histogram(f'{mode}/{s}_{e}_pos_pred', pos_pred[id_lst], global_step=global_step)
                self.writer.add_histogram(f'{mode}/{s}_{e}_neg_pred', neg_pred[id_lst], global_step=global_step)
                self.writer.add_histogram(f'{mode}/{s}_{e}_cat_pred', cat_pred[id_lst], global_step=global_step)

    def get_bins_eid_lst(self, train_g, val_g, test_g, eid_dict):
        # 获取需要切分的边的子图
        train_g = dgl.edge_subgraph(train_g, eid_dict['train'], relabel_nodes=False)
        val_g = dgl.edge_subgraph(val_g, eid_dict['val'], relabel_nodes=False)
        test_g = dgl.edge_subgraph(test_g, eid_dict['test'], relabel_nodes=False)

        if self.task == 'Rate':
            etype = 'rate'
        else:
            etype = 'trust'

        train_e_id_lst = []
        val_e_id_lst = []
        test_e_id_lst = []

        # 分组起点终点
        self.bins_start_end_lst = []
        # u_num_lst = []
        for i in range(len(self.bin_sep_lst)):
            # 获取对应的mask，找到train_g里面在当前度范围内的用户id
            s = self.bin_sep_lst[i]
            if i == len(self.bin_sep_lst) - 1:
                mask = (train_g.out_degrees(etype=etype) >= s)
                e = '∞'
            else:
                e = self.bin_sep_lst[i + 1]
                mask = (train_g.out_degrees(etype=etype) >= s) * \
                       (train_g.out_degrees(etype=etype) < e)
            self.bins_start_end_lst.append((s, e))
            # 所有在这个度范围内的用户id tensor
            u_lst = train_g.nodes('user').masked_select(mask)
            # u_num_lst.append(len(u_lst) / self.user_num)
            # 训练集
            u = train_g.edges(etype=etype)[0]
            # masked select配合isin 把所有在ulst中的u都选出来
            e_id = torch.arange(u.shape[0]).masked_select(torch.isin(u, u_lst))
            train_e_id_lst.append(e_id)
            # 验证集
            u = val_g.edges(etype=etype)[0]
            e_id = torch.arange(u.shape[0]).masked_select(torch.isin(u, u_lst))
            val_e_id_lst.append(e_id)
            # 测试集
            u = test_g.edges(etype=etype)[0]
            e_id = torch.arange(u.shape[0]).masked_select(torch.isin(u, u_lst))
            test_e_id_lst.append(e_id)
        self.metric.bins_id_lst = val_e_id_lst  # 验证集，测试集不用算metric
        tqdm.write(self._log(f'train edges num: {[len(lst) for lst in train_e_id_lst]}'))
        tqdm.write(self._log(f'val edges num: {[len(lst) for lst in val_e_id_lst]}'))
        tqdm.write(self._log(f'test edges num: {[len(lst) for lst in test_e_id_lst]}'))
        return train_e_id_lst, val_e_id_lst, test_e_id_lst

    def get_side_info(self):
        return None

    def construct_negative_graph(self, g, k, etype):
        # 负采样，按用户进行
        src_ntype, edge_type, dst_ntype = etype
        relabeled_src, relabeled_dst = g.edges(etype=etype)
        neg_src = relabeled_src.repeat_interleave(k)
        src = g.nodes[src_ntype].data['_ID'][relabeled_src]

        unique_u, idx_in_u_lst = torch.unique(src, return_inverse=True)
        u_lst = nList(unique_u.tolist())

        map_array = g.nodes[dst_ntype].data['_ID'].detach().cpu().numpy()
        neg_samples = torch.from_numpy(
            subg_neg_sampling(u_lst, self.history_lst[edge_type], map_array, k).reshape(-1, k))
        neg_dst = neg_samples[idx_in_u_lst].reshape(-1).to(g.device)

        return dgl.heterograph(
            {etype: (neg_src, neg_dst)},
            num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        )

    def step(self, mode='train', **inputs):
        # 模型单步计算
        if self.task == 'Rate':
            etype = ('user', 'rate', 'item')
        elif self.task == 'Link':
            etype = ('user', 'trust', 'user')

        if mode == 'train':
            train_pos_g = inputs['graphs'].to(self.device)
            cur_step = inputs['cur_step']
            train_neg_g = self.construct_negative_graph(train_pos_g, self.train_neg_num, etype=etype)
            self.model.train()
            self.optimizer.zero_grad()
            pos_pred, neg_pred = self.model(
                train_pos_g,
                train_pos_g,
                train_neg_g,
            )
            neg_pred = neg_pred.reshape(-1, self.train_neg_num)
            if self.bin_sep_lst is not None and self.is_visulized == True:
                if cur_step == 0:
                    self.log_pred_histgram(pos_pred, neg_pred, mode)
            loss = self.loss_func(pos_pred, neg_pred)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        elif mode == 'evaluate' or mode == 'test':
            with torch.no_grad():
                input_nodes, pos_pred_g, neg_pred_g, blocks = inputs['graphs']
                if isinstance(input_nodes, dict):
                    input_nodes = {k: v.to(self.device) for k, v in input_nodes.items()}
                else:
                    input_nodes = input_nodes.to(self.device)
                blocks = [b.to(self.device) for b in blocks]
                pos_pred_g = pos_pred_g.to(self.device)
                neg_pred_g = neg_pred_g.to(self.device)

                self.model.eval()
                pos_pred, neg_pred = self.model(
                    blocks,
                    pos_pred_g,
                    neg_pred_g,
                    input_nodes
                )
                neg_pred = neg_pred.reshape(-1, self.neg_num)
                if self.bin_sep_lst is not None and self.is_visulized == True:
                    self.log_pred_histgram(pos_pred, neg_pred, mode)
                loss = self.loss_func(pos_pred, neg_pred)
                self.metric.compute_metrics(pos_pred.cpu(), neg_pred.cpu(), task=self.task)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def wrap_step_loop(self, mode, data_loader: dgl.dataloading.DataLoader, side_info, loss_name):
        all_loss_lst = [0.0 for _ in range(len(loss_name))]
        if self.is_log:
            bar_loader = tqdm(enumerate(data_loader), total=len(data_loader))
        else:
            bar_loader = enumerate(data_loader)
        if mode != 'train':
            self.metric.iter_step = 0
        for i, graphs in bar_loader:
            loss_lst = self.step(
                mode=mode, graphs=graphs,
                side_info=side_info, cur_step=i)
            for j in range(len(loss_name)):
                if len(loss_name) == 1:
                    loss_lst = [loss_lst]
                all_loss_lst[j] += loss_lst[j]
            if mode != 'train':
                self.metric.iter_step += 1

        for j in range(len(loss_name)):
            all_loss_lst[j] /= len(data_loader)
        return all_loss_lst

    def _train(self, loss_name, side_info: dict=None):
        # 整体训练流程
        tqdm.write(self._log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])

        side_info = self.get_side_info()
        for e in range(1, epoch + 1):
            """
            write codes for train
            and return loss
            """
            self.cur_e = e

            # # 分箱看histgram
            # if self.bin_sep_lst is not None and self.is_visulized == True:
            #     self.bins_id_lst = self.train_e_id_lst

            # train
            all_loss_lst = self.wrap_step_loop('train', self.train_loader, side_info, loss_name)
            metric_str = f'Train Epoch: {e}\n'
            for j in range(len(loss_name)):
                metric_str += f'{loss_name[j]}: {all_loss_lst[j]:.4f}\t'
            tqdm.write(self._log(metric_str))

            if e % self.eval_step == 0:
                if self.bin_sep_lst is not None:
                    self.bins_id_lst = self.val_e_id_lst
                # 在训练图上跑节点表示，在验证图上预测边的概率
                self.metric.clear_metrics()
                all_loss_lst = self.wrap_step_loop('evaluate', self.val_loader, side_info, loss_name)
                self.metric.get_batch_metrics()
                metric_str = f'Evaluate Epoch: {e}\n'
                for j in range(len(loss_name)):
                    metric_str += f'{loss_name[j]}: {all_loss_lst[j]:.4f}\t'
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
            self.metric.bins_id_lst = self.test_e_id_lst
            self.bins_id_lst = self.test_e_id_lst

        all_loss_lst = self.wrap_step_loop('test', self.test_loader, side_info, loss_name)
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        for j in range(len(loss_name)):
            metric_str += f'{loss_name[j]}: {all_loss_lst[j]:.4f}\t'
        metric_str += '\n'
        metric_str = self._generate_metric_str(metric_str, is_val=False)
        tqdm.write(self._log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)

        test_hr10 = self.metric.metric_dict[self.task]['HR'][10]['value']
        return test_hr10