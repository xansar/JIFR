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
from run import run, get_config, HiddenPrints

def objective(trail):
    config = get_config()
    # 设置需要搜索的参数及搜索空间
    config['OPTIM']['learning_rate'] = str(trail.suggest_float('learning_rate', 1e-4, 1e0, log=True))
    config['OPTIM']['weight_decay'] = str(trail.suggest_float('weight_decay', 1e-4, 1e0, log=True))

    # 调参时不打印训练信息
    config['LOG'] = False
    with HiddenPrints():
        metric_value = run(config)
    return metric_value

if __name__ == '__main__':
    study = optuna.create_study(
        # sampler=optuna.samplers.TPESampler(
        #     n_startup_trials = 5,   # 一开始随机搜索的次数，默认最开始观测值的数量
        #     n_ei_candidates = 24,    # 每一次计算采集函数随机抽取参数组合的数量
        #     multivariate = True,
        # ),   # 贝叶斯调参
        sampler=optuna.samplers.TPESampler(
            **optuna.samplers.TPESampler.hyperopt_parameters()
        ),  # 贝叶斯调参
        direction='maximize',
        study_name='MF'
    )
    study.optimize(objective, n_trials=30)
    print(study.best_params)
