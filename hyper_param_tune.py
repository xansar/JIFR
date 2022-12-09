#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   hyper_param_tune.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/12/3 19:32   zxx      1.0         None
"""

# import lib
import optuna
from optuna.visualization import plot_slice # 独立参数
from optuna.visualization import plot_param_importances # 参数重要性
from optuna.visualization import plot_parallel_coordinate   # 高维度参数关系
from optuna.visualization import plot_intermediate_values   # trial学习曲线
from optuna.visualization import plot_optimization_history  # 优化历史
from optuna.visualization import plot_contour    # 等高线

import os
from run import run, get_config, parse_args, HiddenPrints
import random
import numpy as np
import time
import threading

# 设置全局变量
SEED = 2022  # 占位，会重新抽
ARGS = None
mu = threading.Lock()

def objective(trial):
    global ARGS, EPOCH, SEED
    config = get_config(ARGS)

    # 设置需要搜索的参数及搜索空间
    config['OPTIM']['embedding_learning_rate'] = str(trial.suggest_float('embed_lr', 1e-3, 1e0, log=True))
    config['OPTIM']['embedding_weight_decay'] = str(trial.suggest_float('embed_w', 1e-3, 1e1, log=True))
    if 'mlp_weight_decay' in config['OPTIM'].keys():    # 不对mlp embedding做正则化
        config['OPTIM']['mlp_learning_rate'] = str(trial.suggest_float('mlp_lr', 1e-3, 1e0, log=True))
    if 'lamda_t' in config['OPTIM'].keys():    # trustsvd link loss 系数
        config['OPTIM']['lamda_t'] = str(trial.suggest_float('lamda_t', 1e-2, 1e1, log=True))

    # 重新抽取randomseed，根据这里的采样抽，防止重复
    config['TRAIN']['random_seed'] = str(int(eval(config['OPTIM']['embedding_weight_decay']) * 1561365))
    # random.seed(SEED)

    # 学习率只衰减不重启
    config['OPTIM']['T_0'] = str(15)
    # 调参时不打印训练信息
    config['LOG'] = False
    with HiddenPrints():
        metric_value = run(config, trial=trial)
    return metric_value

def single_search(model_name, n_trials, n_jobs):
    global SEED
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(
            **optuna.samplers.TPESampler.hyperopt_parameters(),
            seed=SEED,
            # 实验功能
            multivariate=True,
            group=True,
            warn_independent_sampling=True,
            constant_liar=True
        ),  # 贝叶斯调参
        direction='maximize',
        study_name=model_name,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study

def visulization(study, model_name, data_name, save_dir=None):
    funcs = [
        'param_importances',
        'parallel_coordinate',
        'intermediate_values',
        'optimization_history',
        'contour',
    ]
    save_pth = os.path.join(save_dir, model_name, data_name)
    # for func in funcs:
    #     fig = eval('plot_' + func)(study)
    #     if not os.path.isdir(save_pth):
    #         os.makedirs(save_pth)
    #     # 保存图片
    #     fig.write_image(os.path.join(save_pth, f'{func}.jpg'), format='jpg')
    # 写入最优参数
    with open(os.path.join(save_pth, f'best_params.txt'), 'w') as f:
        for k, v in study.best_params.items():
            f.write(f'{k}: {v}\n')
            # print(f'{k}: {v}')
        f.write(f'best nDCG: {study.best_value}\n')
        # print(f'best nDCG: {study.best_value}')
        f.close()


def search(model_name, data_name, epoch, train_batch_size, eval_batch_size, n_trials, n_jobs):
    global ARGS
    ARGS = parse_args()
    ARGS.model_name = model_name
    ARGS.data_name = data_name
    ARGS.epoch = epoch
    ARGS.train_batch_size = train_batch_size
    ARGS.eval_batch_size = eval_batch_size

    study = single_search(model_name, n_trials, n_jobs)

    if mu.acquire(True):
        visulization(study, model_name=model_name, data_name=data_name, save_dir='./params_search')
        mu.release()
    # print(
    #     "Remember to clean the save dir after params searching!!!"
    # )

if __name__ == '__main__':
    models = ['DiffnetPP']
    datasets = ['Flickr']
    for model in models:
        for dataset in datasets:
            if not os.path.exists(os.path.join('./log', model, dataset)):
                os.makedirs(os.path.join('./log', model, dataset))
            if not os.path.exists(os.path.join('./save', model, dataset)):
                os.makedirs(os.path.join('./save', model, dataset))
            if not os.path.exists(os.path.join('./params_search', model, dataset)):
                os.makedirs(os.path.join('./params_search', model, dataset))
        search(
            model_name='DiffnetPP',
            data_name='Flickr',
            train_batch_size=10240,
            eval_batch_size=10240,
            epoch=16,   # 每次实验运行的epoch数
            n_trials=3,
            n_jobs=3
        )
    #
    # import os
    # os.system('/root/upload.sh')


