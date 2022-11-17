#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   draw_pic.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/17 13:51   zxx      1.0         None
"""

# import lib
import matplotlib.pyplot as plt
import numpy as np
import re
import os

def get_data_from_log(models_lst, dataset_name="Epinions"):
    data_dict = {}
    processed_data = {}
    for m in models_lst:
        fn_lst = [n + '.txt' for n in os.listdir(f'./log/{m}/{dataset_name}') if 'txt' not in n]
        data_dict[m] = {}
        processed_data[m] = {}
        data_dict[m]['HR@3'] = []
        data_dict[m]['HR@5'] = []
        data_dict[m]['HR@10'] = []
        data_dict[m]['nDCG@3'] = []
        data_dict[m]['nDCG@5'] = []
        data_dict[m]['nDCG@10'] = []
        for fn in fn_lst:
            with open(f'./log/{m}/{dataset_name}/{fn}', 'r', encoding='utf-8') as f:
                data_lst = f.readlines()[-15:]
            datas = [eval(d) for d in re.compile(r'\d\.\d{4}').findall(' '.join(data_lst))]
            # print(datas)
            data_dict[m]['HR@3'].append(datas[:6])
            data_dict[m]['HR@5'].append(datas[6:12])
            data_dict[m]['HR@10'].append(datas[12:18])
            data_dict[m]['nDCG@3'].append(datas[18:24])
            data_dict[m]['nDCG@5'].append(datas[24:30])
            data_dict[m]['nDCG@10'].append(datas[30:])
        for metric in ['HR', 'nDCG']:
            for k in [3, 5, 10]:
                processed_data[m][f'{metric}@{k}'] = {}
                data_array = np.array(data_dict[m][f'{metric}@{k}'])
                assert data_array.shape == (3, 6)
                mean_ = np.mean(data_array, axis=0)
                std_ = np.std(data_array, axis=0, ddof=1)
                processed_data[m][f'{metric}@{k}'].update({'mean': mean_, 'std': std_})
    return processed_data

def draw_grouped_bars(models_lst, processed_data):
    labels = ['total', '0-8', '8-16', '16-32', '32-64', '64-']
    model_name = models_lst
    metric_name = ['HR@3', 'nDCG@3', 'HR@5', 'nDCG@5', 'HR@10', 'nDCG@10']
    plt.style.use('tableau-colorblind10')
    plt.figure(figsize=(35, 20))
    plt.rcParams.update({"font.size": 20})
    for i in range(6):
        metric = metric_name[i]
        plt.subplot(eval(str(32) + str(i + 1)))
        # plt.ylim(0, 0.4)
        x = np.arange(len(labels))
        width = 0.15
        for i in range(5):
            m = model_name[i]
            y = processed_data[m][metric]['mean']
            yerr = processed_data[m][metric]['std']
            if m == 'FusionLightGCN':
                m = 'F-LightGCN'
            plt.bar(x - (i - 2) * width, y, width, label=m, yerr=yerr, error_kw=dict(elinewidth=3, capsize=4))
        plt.ylabel('Scores', fontsize=24)
        plt.title(metric)
        plt.xticks(x, labels=labels, fontsize=24)
        plt.tick_params(labelsize=24)
    # plt.legend(loc='upper right', fontsize=20)
    plt.legend(loc=(1.05, 3), fontsize=20)
    plt.show()

if __name__ == '__main__':
    models_lst = ['TrustSVD', 'DiffnetPP', 'FusionLightGCN', 'LightGCN', 'MF']  # 列表中越靠右，bar越靠左，legend越靠下
    processed_data = get_data_from_log(models_lst, 'Epinions')
    draw_grouped_bars(models_lst, processed_data)
