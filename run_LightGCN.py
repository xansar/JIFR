#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run_LightGCN.py    
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
    parser.add_argument('--config_pth', type=str, default='LightGCN.ini',
                        help='Choose config')
    return parser.parse_args()


def run(config_pth):
    config = get_config(config_pth)
    seed = eval(config['TRAIN']['random_seed'])
    setup_seed(seed)
    data_name = config['DATA']['data_name']
    model_name = config['MODEL']['model_name']
    dataset_name = data_name + 'Rate' + model_name
    # data_pth = '../data/ExtendedEpinions/splited_data/MFModel/val'
    data_pth = os.path.join('./data', data_name, 'splited_data')
    dataset = eval(dataset_name)(data_pth)
    model = eval(model_name + 'Model')(config)
    # model.apply(weight_init)

    # optimizer
    lr = eval(config['OPTIM']['learning_rate'])
    weight_decay = eval(config['OPTIM']['weight_decay'])
    optimizer_name = 'torch.optim.' + config['OPTIM']['optimizer']
    optimizer = eval(optimizer_name)(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    lr_scheduler_name = 'torch.optim.lr_scheduler.' + config['OPTIM']['lr_scheduler']
    lr_adapt_step = eval(config['OPTIM']['lr_adapt_step'])
    lr_scheduler = eval(lr_scheduler_name)(optimizer, lr_adapt_step)
    # loss func
    loss_name = config['LOSS']['loss_name']
    loss_func = eval(loss_name)(reduction='mean')
    # metric
    ks = eval(config['METRIC']['ks'])
    metric_name = eval(config['METRIC']['metric_name'])
    metric = eval(model_name + 'Metric')(ks=ks, metric_name=metric_name)
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
