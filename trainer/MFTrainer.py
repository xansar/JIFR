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


class MFTrainer:
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
        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        self.model_name = self.config['MODEL']['model_name']
        self.log_dir = self.config['TRAIN']['log_pth']
        if not os.path.isdir(os.path.join(self.log_dir, self.model_name)):
            os.mkdir(os.path.join(self.log_dir, self.model_name))
        self.log_pth = os.path.join(self.log_dir, self.model_name, str(self.random_seed) + f'_{self.model_name}.txt')

        # 打印config
        self.print_config()

        self.train_rate_loader = dataloader_dict['train']
        # self.train_link_loader = train_link_loader
        self.val_rate_loader = dataloader_dict['val']
        # self.val_link_loader = val_link_loader
        self.test_rate_loader = dataloader_dict['test']
        # self.test_link_loader = test_link_loader

        self.user_num = eval(config['MODEL']['user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.metric = metric
        with open(os.path.join(f'./data/ExtendedEpinions/splited_data/', 'user2v.json'), 'r') as f:
            self.user2item = json.load(f)['user2item']
        self.neg_num = eval(config['DATA']['neg_num'])


        self.data_name = config['DATA']['data_name']
        self.device = config['TRAIN']['device']
        self.eval_step = eval(config['TRAIN']['eval_step'])
        self.warm_epoch = eval(config['TRAIN']['warm_epoch'])
        self.ks = eval(config['METRIC']['ks'])

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
            rate = inputs['rate'].to(self.device)
            user = inputs['user'].to(self.device)
            item = inputs['item'].to(self.device)
            self.model.train()
            self.optimizer.zero_grad()
            # neg_idx：每一行每个用户对应的neg_item
            # user_idx_single：batch包含的user列表
            # user_id_lst：标注batch内的每个样本对应的负样本的id
            # neg_idx, user_idx_single, user_id_lst = self._negative_sample(user, k=self.neg_num)

            # neg_item = self._get_negative_sample(user, k=1).reshape(-1)
            pos_pred = self.model(
                user=user,
                item=item
            )
            # neg_pred = self.model(
            #     user=user,
            #     item=neg_item
            # )
            loss = self.loss_func(pos_pred, rate)
            # loss = self.loss_func(pos_pred, rate)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            return loss.item()

        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                rate = inputs['rate'].to(self.device)
                user = inputs['user'].to(self.device)
                item = inputs['item'].to(self.device)
                # 一定注意转置和reshape的顺序
                neg_item = self._get_negative_sample(user, k=self.neg_num).t().reshape(-1)
                pos_pred = self.model(
                    user=user,
                    item=item
                )
                # 一定注意转置和reshape的顺序
                neg_pred = self.model(
                    user=user.repeat(self.neg_num),
                    item=neg_item
                ).reshape(self.neg_num, -1).t()
                loss = self.loss_func(pos_pred, rate)
                pos_pred = pos_pred.cpu().reshape(-1, 1)
                neg_pred = neg_pred.cpu()
                self.metric.compute_metrics(pos_pred, neg_pred)
                """
                下面是原始的负采样方式
                """
                # # neg_idx：每一行每个用户对应的neg_item
                # # user_idx_single：batch包含的user列表
                # # user_id_lst：标注batch内的每个样本对应的负样本的id
                # neg_idx, user_idx_single, user_id_lst = self._negative_sample(user, k=self.neg_num)

                # # 每个用户只跟一组负样本计算一次，neg_pred_lst中，pred的组数与不同用户的个数一致
                # neg_pred_lst = []
                # for i in range(len(user_idx_single)):
                #     neg_num = neg_idx[i].shape[0]
                #     cur_user = user_idx_single[i].repeat(neg_num)
                #     item = neg_idx[i]
                #     neg_pred = self.model(user=cur_user, item=item)
                #     neg_pred_lst.append(neg_pred.cpu())
                #
                # # 对batch中的每一个正样本，找一个负样本，这里负样本是较少的
                # # user相同时使用同一组负样本
                # for i in range(pred.shape[0]):
                #     neg_pred_id = user_id_lst[i]
                #     pos_pred = pred[i].reshape(1)
                #     neg_pred = neg_pred_lst[neg_pred_id]
                #     self.metric.compute_metrics(pos_pred, neg_pred)
                return loss.item()
        else:
            raise ValueError("Wrong Mode")

    def _total_negative_sample(self, k):
        # 为所有用户进行负样本抽样
        self.neg_samples = None
        for u in trange(self.user_num, desc='Neg Sampling', leave=True):
            # 当前u的互动列表，item的idx是从0开始计数的
            interacted_item = self.user2item[str(u)]
            mask = torch.ones(self.item_num)    # 0, item_num
            mask[interacted_item] = 0
            # 这是在全局抽100个样本，这里不用加id偏移的原因是模型内部embedding是分开的，分别从0开始计数
            cur_neg = torch.multinomial(mask, k, replacement=True).unsqueeze(0)
            if self.neg_samples is None:
                self.neg_samples = cur_neg
            else:
                self.neg_samples = torch.vstack([self.neg_samples, cur_neg])

    def _get_negative_sample(self, user: torch.Tensor, k):
        return self.neg_samples[user, :k]
        # high = self.neg_samples.shape[1]
        # # 生成一个跟user同等长度，宽度为k的随机idx矩阵
        # idx = torch.randint(0, high, (user.shape[0], k))
        # return self.neg_samples[user].gather(1, idx)

    # def _negative_sample(self, user: torch.Tensor, k):
        # user_idx_single = torch.unique(user, sorted=False)
        # neg_samples = None
        #
        # # 假设test里面，user排序是有序的，相同的id放在一起
        # user_idx_single = torch.unique(user, sorted=False) # 用户的idx，每个只出现一次
        # user_id_lst = []  # 每个用户出现的次数
        # neg_samples = None  # 负采样的idx
        # for idx, u in enumerate(user_idx_single):
        #     # # 当前u出现了几次
        #     num = torch.count_nonzero(user == u)
        #     user_id_lst.extend(num.item() * [idx])
        #     # 当前u的互动列表
        #     interacted_item = self.user2item[str(u.item())]
        #     mask = torch.ones(self.item_num)
        #     mask[interacted_item] = 0
        #     # 这是在全局抽100个样本
        #     cur_neg = torch.multinomial(mask, k, replacement=True).unsqueeze(0)
        #     if neg_samples is None:
        #         neg_samples = cur_neg
        #     else:
        #         neg_samples = torch.vstack([neg_samples, cur_neg])
        # return neg_samples, user_idx_single, user_id_lst


    def _generate_metric_str(self, metric_str):
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

        for e in range(1, epoch + 1):
            all_loss = 0.0
            for idx, rate_data in enumerate(tqdm(self.train_rate_loader)):
                loss = self.step(
                    mode='train',
                    user=rate_data[:, 0].long(),
                    item=rate_data[:, 1].long(),
                    rate=rate_data[:, 2].float()
                )
                all_loss += loss
            all_loss /= idx
            metric_str = f'Train Epoch: {e}\nLoss: {all_loss:.4f}\n'

            if e % self.eval_step == 0:
                self._total_negative_sample(k=self.neg_num)
                self.metric.clear_metrics()
                all_loss = 0.0
                for idx, rate_data in enumerate(tqdm(self.val_rate_loader)):
                    loss = self.step(
                        mode='evaluate',
                        user=rate_data[:, 0].long(),
                        item=rate_data[:, 1].long(),
                        rate=rate_data[:, 2].float(),
                    )
                    all_loss += loss
                all_loss /= idx
                self.metric.get_batch_metrics()
                metric_str += f'Evaluate Epoch: {e}\n'
                metric_str += f'loss: {all_loss:.4f}\n'
                metric_str = self._generate_metric_str(metric_str)

            tqdm.write(self.log(metric_str))
            if self.metric.is_early_stop and e >= self.warm_epoch:
                tqdm.write(self.log("Early Stop!"))
                break
            else:
                self.metric.is_early_stop = False
        tqdm.write(self.log(self.metric.print_best_metrics()))

        self.metric.clear_metrics()
        all_loss = 0.0
        for idx, rate_data in enumerate(tqdm(self.test_rate_loader)):
            loss = self.step(
                mode='evaluate',
                user=rate_data[:, 0].long(),
                item=rate_data[:, 1].long(),
                rate=rate_data[:, 2].float(),
            )
            all_loss += loss
        all_loss /= idx
        self.metric.get_batch_metrics()
        metric_str = f'Test Epoch: \n'
        metric_str += f'loss: {all_loss:.4f}\n'
        metric_str = self._generate_metric_str(metric_str)
        tqdm.write(self.log(metric_str))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)
