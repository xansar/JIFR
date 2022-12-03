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
    config['OPTIM']['learning_rate'] = str(trail.suggest_float('learning_rate', 1e-4, 1e-1, log=True))
    config['OPTIM']['weight_decay'] = str(trail.suggest_float('weight_decay', 1e-4, 1e-1, log=True))

    # 调参时不打印训练信息
    config['LOG'] = False
    with HiddenPrints():
        metric_value = run(config)
    return metric_value

if __name__ == '__main__':
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),   # 贝叶斯调参
        direction='maximize'
    )
    study.optimize(objective, n_trials=5)
    print(study.best_params)
