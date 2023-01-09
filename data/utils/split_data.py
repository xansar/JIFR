#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   split_data.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 19:16   zxx      1.0         None
"""

# import lib
import pandas as pd
import numpy as np
import json
import os
import random
from tqdm import trange

random.seed(1024)
np.random.seed(1024)

def split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, raw_pth='../ExtendedEpinions', model_name='LightGCN'):
    dir_pth = os.path.join(raw_pth, 'splited_data')
    if not os.path.isdir(dir_pth):
        os.mkdir(dir_pth)
    rate_df = pd.read_csv(os.path.join(raw_pth, 'rate_data.csv'), sep=',')
    link_df = pd.read_csv(os.path.join(raw_pth, 'link_data.csv'), sep=',')

    with open(os.path.join(raw_pth, 'behavior_data/user_idx_transfer.json'), 'r') as f:
        user_dict = json.load(f)
        user_num = user_dict['user_num']

    def split_single_lst(lst):
        np.random.shuffle(lst)
        lst_len = len(lst)
        # 这里的规则，测试、验证至少一个，训练中可以完全不出现，从而检测模型对极端冷启动的处理能力
        test_size = max(1, int(test_ratio * lst_len))
        val_size = max(1, int(val_ratio * lst_len))
        train_size = lst_len - val_size - test_size
        # train_size = int(train_ratio * lst_len)
        # res_val_ratio = val_ratio / (val_ratio + test_ratio)
        # val_size = int(res_val_ratio * (lst_len - train_size))

        # # 确保在test有盈余的情况下，val至少有一个
        # test_size = lst_len - train_size - val_size
        # if val_size == 0:
        #     if test_size > 1:
        #         val_size += 1
        #     elif test_size == 1:
        #         if np.random.rand(1) >= 0.5:
        #             val_size += 1

        test_lst = lst[: test_size]
        val_lst = lst[test_size: test_size + val_size]
        train_lst = lst[test_size + val_size: ]
        # train_lst = lst[:train_size]
        # val_lst = lst[train_size: train_size + val_size]
        # test_lst = lst[train_size + val_size:]
        assert len(val_lst) > 0 and len(test_lst) > 0
        return train_lst, val_lst, test_lst

    def write_data(train_, val_, test_, type, mode='w'):
        for m in ['train', 'val', 'test']:
            with open(os.path.join(dir_pth, f'{m}.{type}'), mode) as f:
                # 就是train_...，把里面每一个保存记录的小列表变成字符串
                f.writelines([','.join([str(id) for id in lst]) + '\n' for lst in eval(m + '_')])

    write_data([], [], [], 'rate', 'w')
    write_data([], [], [], 'link', 'w')
    user2item = {}
    user2trust = {}
    rate_gt = {
        'val': [],
        'test': []
    }
    link_gt = {
        'val': [],
        'test': []
    }

    for i in trange(user_num):
        # user2item[i] = rate_df[rate_df.user==i]['item'].values.tolist()
        # user2trust[i] = link_df[link_df.user1==i]['user2'].values.tolist()
        item_lst = rate_df[rate_df.user==i][['user', 'item', 'rate']].values.tolist()
        trust_lst = link_df[link_df.user1==i][['user1', 'user2', 'weight']].values.tolist()

        item_train_lst, item_val_lst, item_test_lst = split_single_lst(item_lst)
        rate_gt['val'].append([l[1] for l in item_val_lst])
        rate_gt['test'].append([l[1] for l in item_test_lst])
        user2item[i] = [l[1] for l in item_train_lst]

        trust_train_lst, trust_val_lst, trust_test_lst = split_single_lst(trust_lst)
        link_gt['val'].append([l[1] for l in trust_val_lst])
        link_gt['test'].append([l[1] for l in trust_test_lst])
        user2trust[i] = [l[1] for l in trust_train_lst]

        write_data(item_train_lst, item_val_lst, item_test_lst, type='rate', mode='a')
        write_data(trust_train_lst, trust_val_lst, trust_test_lst, type='link', mode='a')

    with open(os.path.join(dir_pth, f'user2v.json'), 'w') as f:
        json.dump({'user2item': user2item, 'user2trust':user2trust}, f, indent=2)

    with open(os.path.join(dir_pth, f'gt.json'), 'w') as f:
        json.dump({'rate': rate_gt, 'link': link_gt}, f, indent=2)

if __name__ == '__main__':
    datas = ['Epinions', 'Ciao', 'Yelp', 'Flickr']
    # datas = ['Flickr']
    for data_name in datas:
        split_data(0.8, 0.1, 0.1, raw_pth=f'../{data_name}')






