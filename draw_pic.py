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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import numpy as np
import torch
from scipy.stats import gaussian_kde

import re
import os



def get_data_from_log(models_lst, dataset_name, metrics, ks):
    data_dict = {}
    processed_data = {}
    for m in models_lst:
        # fn_lst = [n + '.txt' for n in os.listdir(f'./log/{m}/{dataset_name}') if 'txt' not in n]
        fn_lst = os.listdir(f'./log/{m}/{dataset_name}')
        data_dict[m] = {}
        processed_data[m] = {}
        for metric in metrics:
            for k in ks:
                data_dict[m][f'{metric}@{k}'] = []
        for fn in fn_lst:
            with open(f'./log/{m}/{dataset_name}/{fn}', 'r', encoding='utf-8') as f:
                data_lst = f.readlines()[-3:]
            # 确定表单的metric顺序
            metrics = [d for d in re.compile(r'[A-Za-z]+').findall(' '.join(data_lst))]
            datas = [eval(d) for d in re.compile(r'\d\.\d{4}').findall(' '.join(data_lst))]
            # print(datas)
            cnt = 0
            for metric in metrics:
                for k in ks:
                    data_dict[m][f'{metric}@{k}'].append(datas[cnt])
                    cnt += 1
            # data_dict[m]['HR@3'].append(datas[:6])
            # data_dict[m]['HR@5'].append(datas[6:12])
            # data_dict[m]['HR@10'].append(datas[12:18])
            # data_dict[m]['nDCG@3'].append(datas[18:24])
            # data_dict[m]['nDCG@5'].append(datas[24:30])
            # data_dict[m]['nDCG@10'].append(datas[30:])
        for metric in metrics:
            for k in ks:
                processed_data[m][f'{metric}@{k}'] = {}
                data_array = np.array(data_dict[m][f'{metric}@{k}'])
                assert data_array.shape == (5,)
                mean_ = np.mean(data_array, axis=0)
                std_ = np.std(data_array, axis=0, ddof=1)
                processed_data[m][f'{metric}@{k}'].update({'mean': mean_, 'std': std_})
    return processed_data

def draw_grouped_bars(models_lst, processed_data, data_name):
    labels = ['total', '0-8', '8-16', '16-32', '32-64', '64-']
    model_name = models_lst
    # metric_name = ['HR@3', 'nDCG@3', 'HR@5', 'nDCG@5', 'HR@10', 'nDCG@10']
    metric_name = ['HR@3', 'HR@5', 'HR@10', 'nDCG@3', 'nDCG@5', 'nDCG@10']
    # xlims = [(0.14, 0.205), (0.18, 0.27), (0.28, 0.355), (0.1, 0.165), (0.13, 0.188), (0.15, 0.215)]   # epinions
    # xlims = [(0.1, 0.185), (0.13, 0.23), (0.17, 0.29), (0.08, 0.155), (0.09, 0.165), (0.11, 0.185)]  # ciao
    plt.style.use('ggplot')
    plt.figure(figsize=(36, 24))
    plt.rcParams.update({"font.size": 20})
    plt.rc('font', family='Times New Roman')
    for i in range(6):
        metric = metric_name[i]
        plt.subplot(eval(str(23) + str(i + 1)))
        # plt.ylim(*xlims[i])
        plt.xlim(-0.75, 7)
        x = np.arange(len(labels))
        width = 0.15
        for i in range(5):
            m = model_name[i]
            y = processed_data[m][metric]['mean']
            yerr = processed_data[m][metric]['std']
            if m == 'FusionLightGCN':
                m = 'F-LightGCN'
            plt.bar(x - (i - 2) * width, y, width, label=m, yerr=yerr, error_kw=dict(elinewidth=1, capsize=4))
        plt.ylabel('Scores', fontsize=34)
        plt.xlabel('Groups', fontsize=34)
        plt.title(metric, fontsize=50, y=1.03)
        plt.xticks(x, labels=labels, fontsize=30)
        plt.tick_params(labelsize=30)
        plt.legend(loc='upper right', fontsize=24)
        # plt.legend(loc=(0.73, 0.35), fontsize=20)
    plt.suptitle(data_name, fontsize=80, y=0.98)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.05, hspace=0.25)
    plt.show()

def draw_grouped_bars_plot(models_lst, processed_data, data_name, y_lst):
    labels = ['total', '0-8', '8-16', '16-32', '32-64', '64-']
    datasets = ['train', 'val', 'test']
    markers = ['s', 'd', 'D', 'o', '*']
    lines = ['-', '-.', '--', '-.', '-']
    model_name = models_lst
    metric_name = ['HR@3', 'HR@5', 'HR@10', 'nDCG@3', 'nDCG@5', 'nDCG@10']
    plt.style.use('seaborn-paper')
    # 获取调色盘
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({"font.size": 20})
    plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(36, 24))
    for i in range(6):
        metric = metric_name[i]
        x = np.arange(len(labels))
        width = 0.25
        ax1 = fig.add_subplot(eval(str(23) + str(i + 1)))
        # 画数据集边数分布图
        for i in range(3):
            y = np.log10([sum(y_lst[i])] + y_lst[i])
            ax1.bar(x - (i - 1) * width, y, width, label=datasets[i], alpha=0.5, color=colors[i])
        plt.legend(loc='upper left', fontsize=24)
        plt.xticks(x, labels=labels, fontsize=30)
        ax1.tick_params(axis='y',labelsize=25)  # y轴字体大小设置
        ax1.set_ylabel('Num of edges(log)', fontsize=34)
        # 画指标折线图
        ax2 = ax1.twinx()
        for i in range(5):
            m = model_name[i]
            y = processed_data[m][metric]['mean']
            yerr = processed_data[m][metric]['std']
            if m == 'FusionLightGCN':
                m = 'F-LightGCN'
            if i > 2:
                color = colors[3]
            else:
                color = colors[4]
            ax2.errorbar(x, y, label=m, yerr=yerr,
                         linestyle=lines[i], color=color, marker=markers[i],
                         markersize=15, linewidth=4, capsize=5, capthick=3)
        ax2.set_ylabel('Scores', fontsize=34)
        ax2.set_xlabel('Groups', fontsize=34)
        ax2.tick_params(axis='y', labelsize=25)
        plt.title(metric, fontsize=50, y=1.03)
        # plt.xticks(x, labels=labels, fontsize=30)
        # plt.tick_params(labelsize=30)
        plt.legend(loc='upper right', fontsize=24)
        # plt.legend(loc=(0.73, 0.35), fontsize=20)
    plt.suptitle(data_name, fontsize=80, y=0.98)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.05, hspace=0.25, wspace=0.25)
    plt.show()

def draw_bars(models_lst, processed_data, data_name):
    width = 0.25
    x = np.arange(len())
    datasets = ['train', 'val', 'test']
    labels = ['0-8', '8-16', '16-24', '24-32', '32-40', '40-48', '48-56', '56-64', '64-72', '72-80', '80-88', '88-96',
              '96-128', '128-256', '256-']
    for i in range(3):
        y = np.log10(y_lst[i])
        plt.bar(x - (i - 2) * width, y, width, label=datasets[i])
    plt.xticks(x, labels=labels, rotation=320)
    plt.title("Ciao")
    plt.legend()
    plt.show()

def draw_dist(g, dataset_name):
    # 获取节点度数
    x = (torch.log(g.out_degrees(etype='rate')) / torch.log(torch.tensor(2))).numpy()
    y = (torch.log(g.out_degrees(etype='trust')) / torch.log(torch.tensor(2))).numpy()
    xy = np.vstack([x, y])
    # 拟合二维高斯分布
    z = gaussian_kde(xy)(xy)
    fig = plt.figure(figsize=(10, 8))
    # 刻度的数量
    grid_num = 12

    plt.style.use('seaborn-paper')
    plt.rc('font', family='Times New Roman')
    # 用来放置散点图和边缘分布图
    gs = GridSpec(4, 4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3])
    ax_marg_y = fig.add_subplot(gs[1:4, 3])

    # 绘制散点图
    ax_joint.scatter(x, y, c=z, marker='o', cmap='Spectral')
    ax_joint.grid()
    # 用来加上标
    sub_map = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
    ax_joint.set_xticks(range(grid_num))
    ax_joint.set_xticklabels(['2' + str(i).translate(sub_map) for i in range(grid_num)], fontsize=15)
    ax_joint.set_yticks(range(grid_num))
    ax_joint.set_yticklabels(['2' + str(i).translate(sub_map) for i in range(grid_num)], fontsize=15)
    # 控制刻度最大最小值
    ax_joint.set_xlim(left=0., right=grid_num - 1)
    ax_joint.set_ylim(bottom=0., top=grid_num - 1)

    # x边缘分布
    n, bins, patches = ax_marg_x.hist(x, bins=range(grid_num), density=True, alpha=0.5, align='left')
    # 拟合
    marg_x_p = gaussian_kde(x)(bins)
    ax_marg_x.plot(bins, marg_x_p, '--')  # 绘制y的曲线
    # y边缘分布
    n, bins, patches = ax_marg_y.hist(y, orientation="horizontal", density=True, alpha=0.5, align='left')
    # 拟合
    marg_y_p = gaussian_kde(y)(bins)
    ax_marg_y.plot(marg_y_p, bins, '--')  # 绘制y的曲线

    # 设置边缘分布刻度
    ax_marg_x.set_xticks(range(grid_num))
    ax_marg_x.set_xticklabels(range(grid_num), fontsize=15)
    ax_marg_x.set_xlim(left=0., right=grid_num - 1)
    ax_marg_y.set_yticks(range(grid_num))
    ax_marg_y.set_yticklabels(range(grid_num), fontsize=15)
    ax_marg_y.set_ylim(bottom=0., top=grid_num - 1)
    # 不显示边缘分布刻度标签
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel('Rate Degree', fontsize=20)
    ax_joint.set_ylabel('Trust Degree', fontsize=20)

    # Set labels on marginals
    ax_marg_y.set_xlabel('Rate Degree Density', fontsize=10)
    ax_marg_x.set_ylabel('Trust Degree Density', fontsize=10)
    norm = mcolors.Normalize(vmin=np.min(z), vmax=np.max(z))
    # 前面是值到颜色的映射，后面表示在那个子图的位置放
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='Spectral'))
    fig.suptitle(dataset_name, fontsize=40)
    plt.show()


if __name__ == '__main__':
    print(plt.style.available)
    models_lst = ['TrustSVD', 'DiffnetPP', 'FLGN', 'LightGCN', 'MF', 'SocialLGN']  # 列表中越靠右，bar越靠左，legend越靠下
    # models_lst = ['FNCL', 'NCL']  # 列表中越靠右，bar越靠左，legend越靠下
    data_name = 'Ciao'
    metrics = ['nDCG', 'Recall', 'Precision']
    ks = [10, 20, 50]
    processed_data = get_data_from_log(models_lst, data_name, metrics, ks)
    print(processed_data)

    metrics_at_k = processed_data['MF'].keys()
    info_table = [['Ciao'] + [m for m in metrics_at_k]]
    from tabulate import tabulate
    for model in processed_data.keys():
        info_table.append([model])
        for metric in metrics_at_k:
            mean = processed_data[model][metric]['mean']
            std = processed_data[model][metric]['std']
            value_str = f'{mean:.4f}±{std:.4f}'
            info_table[-1].append(value_str)
    print(tabulate(info_table))

    # y_lst = {
    #     'Yelp': [
    #         [26855, 34889, 31901, 17547, 8873],
    #         [4757, 4271, 3946, 2174, 1104],
    #         [4751, 6125, 4796, 2428, 1163],
    #     ],
    #     'Flickr': [
    #         [9106, 17915, 33129, 44235, 132868],
    #         [1763, 2211, 4123, 5518, 16594],
    #         [1778, 3101, 4916, 6038, 17073],
    #     ],
    #     'Epinions': [
    #         [25350, 45149, 68707, 66700, 86678],
    #         [4935, 5543, 8532, 8315, 10823],
    #         [5038, 7811, 10210, 9140, 11219],
    #     ],
    #     'Ciao': [
    #         [6064, 14428, 25586, 27983, 52351],
    #         [1069, 1778, 3183, 3483, 6540],
    #         [1103, 2514, 3816, 3829, 6752],
    #     ]
    # }
    # draw_grouped_bars_plot(models_lst, processed_data, data_name, y_lst[data_name])

