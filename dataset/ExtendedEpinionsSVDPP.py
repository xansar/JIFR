#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ExtendedEpinionsTrustSVD.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/18 18:10   zxx      1.0         None
"""

# import lib
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import json

class ExtendedEpinionsRateSVDPP(Dataset):
    def __init__(self, data_pth, config):
        self.config = config
        self.user_num = eval(config['MODEL']['pred_user_num'])
        self.item_num = eval(config['MODEL']['item_num'])
        self.total_user_num = eval(config['MODEL']['total_user_num'])
        self.num_sample = eval(config['MODEL']['num_sample'])
        self.model_name = config['MODEL']['model_name']
        self.data_pth = data_pth

        dir_name, base_name = os.path.split(self.data_pth)
        self.mode = base_name
        self.dir_pth = dir_name
        self.aid_data_pth = os.path.join(self.dir_pth, self.model_name)
        if not os.path.isdir(self.aid_data_pth):
            os.mkdir(self.aid_data_pth)

        self.read_data()
        self.inverted_list()
        super(ExtendedEpinionsRateSVDPP, self).__init__()

        self.need_sample_user_id = None

    def sample_neighbours(self):
        # 当用户评价过的物品太多，或者好友太多的时候，模型计算很慢
        # 考虑在每个epoch前，为所有用户采样一次，这个epoch内，用户的历史物品和好友不变
        self.cur_user2item = {}
        self.cur_user2trust = {}
        if self.need_sample_user_id is None:
            self.need_sample_user_id = range(self.user_num)

        for u in self.need_sample_user_id:
            user2item_lst = self.user2item.get(str(u), [])
            num_sample = min(self.num_sample, len(user2item_lst))
            self.cur_user2item[str(u)] = np.random.choice(user2item_lst, num_sample, replace=False)

            user2trust_lst = self.user2trust.get(str(u), [])
            num_sample = min(self.num_sample, len(user2trust_lst))
            self.cur_user2trust[str(u)] = np.random.choice(user2trust_lst, num_sample, replace=False)

    def __getitem__(self, idx):
        ## 用户的历史item互动记录
        ## 用户的好友记录
        ## 评分
        # 用户、物品
        u, i = self.rate_u_i[idx]
        # 评分
        score = self.rate_score[idx]
        # 用户评价过的物品
        # 这里如果不采样，太多了
        rated_items = self.cur_user2item.get(str(u), [])
        # 用户评价过的物品的总数
        items_num = len(self.user2item.get(str(u), []))
        # 评价过物品i的用户数量
        user_rate_i_num = len(self.item2user.get(str(i), []))
        # 用户的信任列表
        trusts = self.cur_user2trust.get(str(u), [])
        # 用户的信任列表长度
        trusts_num = len(self.user2trust.get(str(u), []))
        # 信任用户u的数量
        user_trust_u_num = len(self.trust2user.get(str(u), []))
        return {
            'users': u,
            'items': i,
            'scores': score,
            'rated_items': rated_items,
            'items_nums': items_num,
            'user_rate_i_num': user_rate_i_num,
            'trusts': trusts,
            'trusts_nums': trusts_num,
            'user_trust_u_num': user_trust_u_num
        }

    def __len__(self):
        return len(self.rate_score)

    def read_data(self):
        rate_data_pth = self.data_pth + '.rate'
        link_data_pth = self.data_pth + '.link'

        rate = np.loadtxt(rate_data_pth, delimiter=',')
        link = np.loadtxt(link_data_pth, delimiter=',')
        # 切分记录与评分
        self.rate_u_i = rate[:, :2].astype(np.int32).tolist()
        self.rate_score = rate[:, 2].astype(np.float32).tolist()
        self.normalize(5, 0)
        self.link_u_v = link[:, :2].astype(np.int32).tolist()
        self.link_score = link[:, 2].astype(np.float32).tolist()

        f = open(os.path.join(self.dir_pth, 'user2v.json'), 'r')
        self.user2history = json.load(f)['user2item']
        f.close()

    def normalize(self, max=None, min=None):
        if max is None:
            max = np.max(self.rate_score)
        if min is None:
            min = np.min(self.rate_score)
        self.rate_score = ((np.array(self.rate_score) - min) / (max - min)).tolist()

    def inverted_list(self):
        # 读取倒排表？好像是正排表
        inverted_list_pth = os.path.join(self.aid_data_pth, 'inverted_list.json')

        if os.path.exists(inverted_list_pth):
            print('=============read process inverted list==============')
            with open(inverted_list_pth, 'r') as f:
                inverted_lists = json.load(f)
                self.user2item = inverted_lists['user2item']
                self.user2trust = inverted_lists['user2trust']
                self.item2user = inverted_lists['item2user']
                self.trust2user = inverted_lists['trust2user']
        else:
            print('=============begin process inverted list==============')
            self.user2item = {}
            self.item2user = {}
            self.user2trust = {}
            self.trust2user = {}
            for record in self.rate_u_i:
                u, i = int(record[0]), int(record[1])
                if str(u) in self.user2item.keys():
                    self.user2item[str(u)].append(i)
                else:
                    self.user2item[str(u)] = [i]

                if str(i) in self.item2user.keys():
                    self.item2user[str(i)].append(u)
                else:
                    self.item2user[str(i)] = [u]

            for record in self.link_u_v:
                u, v = int(record[0]), int(record[1])
                if str(u) in self.user2trust.keys():
                    self.user2trust[str(u)].append(v)
                else:
                    self.user2trust[str(u)] = [v]

                if str(v) in self.trust2user.keys():
                    self.trust2user[str(v)].append(u)
                else:
                    self.trust2user[str(v)] = [u]

            with open(inverted_list_pth, 'w') as f:
                json.dump({
                    'user2item': self.user2item,
                    'user2trust': self.user2trust,
                    'item2user': self.item2user,
                    'trust2user': self.trust2user
                }, f)
            print('=============end process inverted list==============')

def TrustSVD_collate_fn(data_lst):
    bsz = len(data_lst)
    users = torch.zeros(bsz, dtype=torch.long)
    items = torch.zeros(bsz, dtype=torch.long)
    scores = torch.zeros(bsz, dtype=torch.float32)
    rated_items = []
    items_nums = torch.zeros(bsz, dtype=torch.float32)
    user_rate_i_num = torch.zeros(bsz, dtype=torch.float32)
    trusts = []
    trusts_nums = torch.zeros(bsz, dtype=torch.float32)
    user_trust_u_num = torch.zeros(bsz, dtype=torch.float32)

    for i, data in enumerate(data_lst):
        users[i] = data['users']
        items[i] = data['items']
        scores[i] = data['scores']
        rated_items.append(torch.tensor(data['rated_items'], dtype=torch.long))
        items_nums[i] = data['items_nums']
        user_rate_i_num[i] = data['user_rate_i_num']
        trusts.append(torch.tensor(data['trusts'], dtype=torch.long))
        trusts_nums[i] = data['trusts_nums']
        user_trust_u_num[i] = data['user_trust_u_num']

    rated_items = pad_sequence(rated_items, batch_first=True, padding_value=23143)
    trusts = pad_sequence(trusts, batch_first=True, padding_value=12770)

    res_dict = {}
    for k in data_lst[0].keys():
        res_dict[k] = eval(k)
    return res_dict

if __name__ == '__main__':
    dataset = ExtendedEpinionsRateTrustSVD()
    print(np.mean(dataset.rate_score))
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for x in dataloader:
        print(x)
        break