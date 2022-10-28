#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   transfer_data.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/24 15:52   zxx      1.0         None
"""

# import lib
import pandas as pd
import numpy as np
import json

def transfer_data():
    print('begin transfer')
    link_names = ['user1', 'user2', 'weight', 'date']
    link_df = pd.read_csv('./raw_data/user_rating.txt', delimiter='\t', names=link_names)

    rate_names = ['item', 'user', 'rate', 'date']
    rate_df = pd.read_csv('./raw_data/rating.txt', delimiter='\t', names=rate_names, usecols=[0, 1, 2, 4])
    link_df = link_df[link_df.weight==1]
    link_df.to_csv('./behavior_data/link.csv', index=False, header=True)
    rate_df.to_csv('./behavior_data/rate.csv', index=False, header=True)
    print('end transfer')


def filter_data(threshold=100):
    print('begin filter')

    rate_df = pd.read_csv('./behavior_data/rate.csv', delimiter=',')
    link_df = pd.read_csv('./behavior_data/link.csv', delimiter=',')

    def continue_loop(rate_df_filter, link_df_filter):
        flag_1 = (rate_df_filter['user'].value_counts() < threshold).sum()
        flag_2 = (rate_df_filter['item'].value_counts() < threshold).sum()
        flag_3 = (link_df_filter['user1'].value_counts() < threshold).sum()
        flag_4 = (~rate_df_filter.user.isin(link_df_filter.user1)).sum()
        flag_5 = (~link_df_filter.user1.isin(rate_df_filter.user)).sum()

        for i in range(1, 6):
            print(eval(f'flag_{i}'), end='\t')
        print('')
        return flag_1 > 1 or flag_2 > 1 or flag_3 > 1 or flag_4 > 1 or flag_5 > 1

    def single_process(rate_df_filter, link_df_filter):
        # 所有用户至少有四个邻居
        link_df_filter = link_df_filter[link_df_filter.groupby('user1').user1.transform('count') >= threshold]

        # 将rate中user，item出现少于四次的过滤掉
        rate_df_filter = rate_df_filter[rate_df_filter.groupby('user').user.transform('count') >= threshold]
        rate_df_filter = rate_df_filter[rate_df_filter.groupby('item').item.transform('count') >= threshold]

        # 将在rate user表，不在link user表的过滤掉
        rate_df_filter = rate_df_filter[rate_df_filter.user.isin(link_df_filter.user1)]

        # 将在link user表，不在rating user表的过滤掉
        link_df_filter = link_df_filter[link_df_filter.user1.isin(rate_df_filter.user)]

        return rate_df_filter, link_df_filter

    flag = continue_loop(rate_df, link_df)
    n = 0
    while flag:
        rate_df, link_df = single_process(rate_df, link_df)
        flag = continue_loop(rate_df, link_df)
        n += 1
        if n >= 100:
            break
    print(n)
    link_df.to_csv('./behavior_data/filtered_link.csv', index=False, header=True)
    rate_df.to_csv('./behavior_data/filtered_rate.csv', index=False, header=True)
    print(rate_df.describe())
    print(link_df.describe())
    print('end filter')


def relabel():
    print('begin relabel')
    rate_df = pd.read_csv('./behavior_data/filtered_rate.csv', sep=',')
    link_df = pd.read_csv('./behavior_data/filtered_link.csv', sep=',')

    # 保证rating和link里面的user一致
    assert (~rate_df.user.value_counts().sort_index().index == link_df.user1.value_counts().sort_index().index).sum() == 0

    # 新id
    item_N = len(rate_df.item.value_counts())
    i_array = np.arange(item_N)

    user_N = len(rate_df.user.value_counts())
    u_array = np.arange(user_N)

    # 生成字典
    sorted_user_idx_lst = sorted(rate_df.user.value_counts().index)
    # 只有小于这个数的用户才需要预测
    pred_user_max = len(sorted_user_idx_lst)

    user_raw_idx2new_idx = {}
    user_new_idx2raw_idx = {}
    for i in range(len(sorted_user_idx_lst)):
        raw_id = sorted_user_idx_lst[i]
        new_id = int(u_array[i])
        # print(type(raw_id), type(new_id))
        user_raw_idx2new_idx[raw_id] = new_id
        user_new_idx2raw_idx[new_id] = raw_id
        # print(user_raw_idx2new_idx)
    # 把link里面user2的也放进去
    sorted_user_idx_lst = sorted(link_df.user2.value_counts().index)
    cnt = 0
    for i in range(len(sorted_user_idx_lst)):
        raw_id = sorted_user_idx_lst[i]
        if raw_id not in user_raw_idx2new_idx.keys():
            new_id = cnt + len(u_array)
            cnt += 1
            # print(type(raw_id), type(new_id))
            user_raw_idx2new_idx[raw_id] = new_id
            user_new_idx2raw_idx[new_id] = raw_id
            # print(user_raw_idx2new_idx)

    # 保存
    user_idx_dict = {
        'pred_user_max': pred_user_max,
        'total_user_num': len(user_new_idx2raw_idx),
        'raw2new': user_raw_idx2new_idx,
        'new2raw': user_new_idx2raw_idx
    }
    with open('./behavior_data/user_idx_transfer.json', 'w') as f:
        json.dump(user_idx_dict, f, indent=2)

    # 生成字典
    sorted_item_idx_lst = sorted(rate_df.item.value_counts().index)

    item_raw_idx2new_idx = {}
    item_new_idx2raw_idx = {}
    for i in range(len(sorted_item_idx_lst)):
        raw_id = sorted_item_idx_lst[i]
        new_id = int(i_array[i])
        item_raw_idx2new_idx[raw_id] = new_id
        item_new_idx2raw_idx[new_id] = raw_id

    # 保存
    item_idx_dict = {
        'raw2new': item_raw_idx2new_idx,
        'new2raw': item_new_idx2raw_idx
    }
    with open('./behavior_data/item_idx_transfer.json', 'w') as f:
        json.dump(item_idx_dict, f, indent=2)

    # 替换原有数据
    new_rating_df = rate_df.copy()
    new_rating_df['user'] = new_rating_df['user'].apply(lambda x: user_raw_idx2new_idx[x])
    new_rating_df['item'] = new_rating_df['item'].apply(lambda x: item_raw_idx2new_idx[x])
    new_rating_df.to_csv('./rate_data.csv', index=False, header=True)

    new_link_df = link_df.copy()
    new_link_df['user1'] = new_link_df['user1'].apply(lambda x: user_raw_idx2new_idx[x])
    new_link_df['user2'] = new_link_df['user2'].apply(lambda x: user_raw_idx2new_idx[x])
    new_link_df.to_csv('./link_data.csv', index=False, header=True)

    print('end relabel')


if __name__ == '__main__':
    # transfer_data()
    # filter_data(threshold=60)
    relabel()