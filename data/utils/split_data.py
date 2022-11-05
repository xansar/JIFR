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

def split_data(train_ratio=0.75, val_ratio=0.05, test_ratio=0.2, raw_pth='../ExtendedEpinions', model_name='LightGCN'):
    dir_pth = os.path.join(raw_pth, 'splited_data')
    if not os.path.isdir(dir_pth):
        os.mkdir(dir_pth)
    rate_df = pd.read_csv(os.path.join(raw_pth, 'rate_data.csv'), sep=',')
    link_df = pd.read_csv(os.path.join(raw_pth, 'link_data.csv'), sep=',')

    with open(os.path.join(raw_pth, 'behavior_data/user_idx_transfer.json'), 'r') as f:
        user_dict = json.load(f)
        pred_user_max = user_dict['pred_user_max']

    def split_single_lst(lst):
        np.random.shuffle(lst)
        lst_len = len(lst)
        train_size = int(train_ratio * lst_len)
        val_size = int(val_ratio * lst_len)
        train_lst = lst[:train_size]
        val_lst = lst[train_size: train_size + val_size]
        test_lst = lst[train_size + val_size:]
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
    for i in trange(pred_user_max):
        user2item[i] = rate_df[rate_df.user==i]['item'].values.tolist()
        user2trust[i] = link_df[link_df.user1==i]['user2'].values.tolist()
        item_lst = rate_df[rate_df.user==i][['user', 'item', 'rate']].values.tolist()
        trust_lst = link_df[link_df.user1==i][['user1', 'user2', 'weight']].values.tolist()
        # item_train_lst, item_val_lst, item_test_lst = split_single_lst(item_lst)
        # trust_train_lst, trust_val_lst, trust_test_lst = split_single_lst(trust_lst)
        write_data(*split_single_lst(item_lst), type='rate', mode='a')
        write_data(*split_single_lst(trust_lst), type='link', mode='a')
    with open(os.path.join(dir_pth, f'user2v.json'), 'w') as f:
        json.dump({'user2item': user2item, 'user2trust':user2trust}, f, indent=2)

if __name__ == '__main__':
    split_data(0.8, 0.1, 0.1, raw_pth='../Epinions')






