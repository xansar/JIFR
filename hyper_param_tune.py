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

# 设置全局变量
SEED = 2022  # 占位，会重新抽
EPOCH = 1
ARGS = None

def objective(trial):
    global ARGS, EPOCH, SEED
    config = get_config(ARGS)
    # 统一设置epoch为50，重新抽取randomseed
    config['TRAIN']['epoch'] = str(EPOCH)
    SEED = random.randint(0, 999999)
    config['TRAIN']['random_seed'] = str(SEED)
    random.seed(SEED)
    # 设置需要搜索的参数及搜索空间
    config['OPTIM']['embedding_learning_rate'] = str(trial.suggest_float('embedding_learning_rate', 1e-3, 1e0, log=True))
    config['OPTIM']['embedding_weight_decay'] = str(trial.suggest_float('embedding_weight_decay', 1e-3, 1e1, log=True))
    if 'mlp_weight_decay' in config['OPTIM'].keys():    # 不对mlp embedding做正则化
        config['OPTIM']['mlp_learning_rate'] = str(trial.suggest_float('mlp_learning_rate', 1e-3, 1e0, log=True))

    # 学习率只衰减不重启
    config['OPTIM']['T_0'] = str(config['TRAIN']['epoch'])
    # 调参时不打印训练信息
    config['LOG'] = False
    with HiddenPrints():
        metric_value = run(config, trial=trial)
    return metric_value

def single_search(model_name, n_trials):
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
    study.optimize(objective, n_trials=n_trials)
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
    for func in funcs:
        fig = eval('plot_' + func)(study)
        if not os.path.isdir(save_pth):
            os.makedirs(save_pth)
        # 保存图片
        fig.write_image(os.path.join(save_pth, f'{func}.jpg'), format='jpg')
    # 写入最优参数
    with open(os.path.join(save_pth, 'best_params.txt'), 'w') as f:
        for k, v in study.best_params.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}')
        f.write(f'best nDCG: {study.best_value}\n')
        print(f'best nDCG: {study.best_value}')

def search(model_name, data_name, n_trials):
    global ARGS
    ARGS = parse_args()
    ARGS.model_name = model_name
    ARGS.data_name = data_name

    study = single_search(model_name, n_trials)
    visulization(study, model_name=model_name, data_name=data_name, save_dir='./params_search')
    print(
        "Remember to clean the save dir after params searching!!!"
    )

if __name__ == '__main__':
    search('LightGCN', 'Epinions', 2)


