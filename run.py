#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/26 16:22   zxx      1.0         None
"""

import os

import torch
import numpy as np

import argparse
import random

from dataset import *
from metric import *
from model import *
from trainer import *

from configparser import ConfigParser

CommonModel = ['MF']
GCNModel = ['LightGCN', 'TrustSVD', 'SVDPP', 'Sorec', 'MutualRec', 'FusionLightGCN', 'DiffnetPP', 'GraphRec', 'SocialMF']

use_common_datset = ['LightGCN', 'MF']
use_social_dataset = ['MutualRec', 'FusionLightGCN', 'DiffnetPP', 'GraphRec']
use_directed_social_dataset = ['TrustSVD', 'SVDPP', 'Sorec', 'SocialMF']

class MyConfigParser(ConfigParser):
    def __init__(self, defaults=None):
        super(MyConfigParser, self).__init__()

    def optionxform(self, optionstr):
        return optionstr


def get_config(config_pth):
    config = MyConfigParser()
    config.read('./config/' + config_pth, encoding='utf-8')
    return config._sections


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    # Parses the arguments.
    parser = argparse.ArgumentParser(description="Run Model.")
    parser.add_argument('--config_pth', type=str, default='MF.ini',
                        help='Choose config')
    return parser.parse_args()


def run(config_pth):
    config = get_config(config_pth)
    seed = eval(config['TRAIN']['random_seed'])
    setup_seed(seed)
    data_name = config['DATA']['data_name']
    model_name = config['MODEL']['model_name']
    task = config['TRAIN']['task']

    if model_name in use_common_datset:
        dataset = RateCommonDataset(config)
    else:
        if model_name in use_social_dataset:
            dataset = SocialDataset(config)
        elif model_name in use_directed_social_dataset:
            dataset = SocialDataset(config, directed=True)
        else:
            raise ValueError("Wrong Model Name!!!")

    if model_name in CommonModel:
        model = eval(model_name + 'Model')(config)
    elif model_name in GCNModel:
        etype = dataset[0].etypes
        model = eval(model_name + 'Model')(config, etype)
    else:
        raise ValueError("Wrong Model Name!!!")
    # model.apply(weight_init)

    # optimizer
    lr = eval(config['OPTIM']['learning_rate'])
    weight_decay = eval(config['OPTIM']['weight_decay'])
    optimizer_name = 'torch.optim.' + config['OPTIM']['optimizer']
    optimizer = eval(optimizer_name)(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    lr_scheduler_name = 'torch.optim.lr_scheduler.' + config['OPTIM']['lr_scheduler']
    T_0 = eval(config['OPTIM']['T_0'])  # 学习率第一次重启的epoch数
    T_mult = eval(config['OPTIM']['T_mult'])    # 学习率衰减epoch数变化倍率
    lr_scheduler = eval(lr_scheduler_name)(optimizer, T_0=T_0, T_mult=T_mult, verbose=True)
    # loss func
    loss_name = config['LOSS']['loss_name']
    loss_func = eval(loss_name)(reduction='mean')
    # metric

    metric = BaseMetric(config)
    # trainer
    trainer = eval(model_name + 'Trainer')(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metric=metric,
        dataset=dataset,
        config=config
    )
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    run(args.config_pth)
