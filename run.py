#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/26 16:22   zxx      1.0         None
"""
import sys
import os
import json

import torch
import numpy as np

import argparse
import random

from trainer import *

from configparser import ConfigParser

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class MyConfigParser(ConfigParser):
    def __init__(self, defaults=None):
        super(MyConfigParser, self).__init__()

    def optionxform(self, optionstr):
        return optionstr

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    # Parses the arguments.
    # MODEL
    parser = argparse.ArgumentParser(description="Run Model.")
    parser.add_argument('-m', '--model_name', type=str, default='MF',
                        help='Choose config')
    # DATA
    parser.add_argument('-d', '--data_name', type=str, default=None,
                        help='choose dataset')
    parser.add_argument('-tng', '--train_neg_num', type=int, default=None,
                        help='train neg num')
    parser.add_argument('-ng', '--neg_num', type=int, default=None,
                        help='eval neg num')
    parser.add_argument('-tb', '--train_batch_size', type=int, default=None,
                        help='train batch size')
    parser.add_argument('-vb', '--eval_batch_size', type=int, default=None,
                        help='eval batch size')
    parser.add_argument('-nw', '--num_workers', type=int, default=None,
                        help='num workers')

    # OPTIM
    parser.add_argument('-lr', '--mlp_learning_rate', type=float, default=None,
                        help='mlp learning rate')
    parser.add_argument('-elr', '--embedding_learning_rate', type=float, default=None,
                        help='embed learning rate')
    parser.add_argument('-w', '--mlp_weight_decay', type=float, default=None,
                        help='mlp weight decay')
    parser.add_argument('-ew', '--embedding_weight_decay', type=float, default=None,
                        help='embed weight_decay')
    parser.add_argument('-T', '--T_0', type=int, default=None,
                        help='lr scheduler param T_0')
    parser.add_argument('-Tm', '--T_m', type=int, default=None,
                        help='lr scheduler param T_m')

    # TRAIN
    parser.add_argument('-t', '--task', type=str, default=None,
                        help='choose task')
    parser.add_argument('-e', '--epoch', type=int, default=None,
                        help='epoch')
    parser.add_argument('-estp', '--eval_step', type=int, default=None,
                        help='eval step')
    parser.add_argument('-r', '--random_seed', type=int, default=None,
                        help='random seed')

    # other
    parser.add_argument('-v', '--visulize', type=bool, default=False,
                        help='whether to visulize train logs with tensorboard')
    parser.add_argument('-l', '--log', type=bool, default=True,
                        help='whether to print and save train logs (suggest to False when tune params)')
    return parser.parse_args()

def get_config(args):
    config = MyConfigParser()
    model = args.model_name
    config.read('./config/' + model + '.ini', encoding='utf-8')
    config = config._sections

    arg_class = {
        'data_name': 'DATA',
        'train_neg_num': 'DATA',
        'neg_num': 'DATA',
        'train_batch_size': 'DATA',
        'eval_batch_size': 'DATA',

        'T_0': 'OPTIM',
        'T_mult': 'OPTIM',
        'early_stop_num': 'OPTIM',
        'embedding_learning_rate': 'OPTIM',
        'mlp_learning_rate': 'OPTIM',
        'embedding_weight_decay': 'OPTIM',
        'mlp_weight_decay': 'OPTIM',

        'task': 'TRAIN',
        'epoch': 'TRAIN',
        'eval_step': 'TRAIN',
        'random_seed': 'TRAIN'
    }

    skip_show_arg_lst = ['model_name', 'visulize', 'log']
    for arg in vars(args):
        if arg in skip_show_arg_lst:
            continue
        else:
            arg_value = getattr(args, arg)
            if arg_value is not None:
                print(f'cmd change arg {arg}: config_value-{config[arg_class[arg]][arg]}, new_value-{str(arg_value)}')
                config[arg_class[arg]][arg] = str(arg_value)


    if args.data_name is None:
        data_name = config['DATA']['data_name']
    else:
        data_name = args.data_name

    data_info = get_data_info(data_name)
    config['DATA']['data_name'] = data_name
    config['MODEL'].update(data_info)

    config.update({'VISUALIZED': args.visulize})
    config.update({'LOG': args.log})
    return config

def get_data_info(data_name):
    fp = os.path.join('./data/', data_name, 'behavior_data/data_info.json')
    with open(fp, 'r') as f:
        data_info = json.load(f)
    return data_info

def run(config, trial=None):
    seed = eval(config['TRAIN']['random_seed'])
    setup_seed(seed)

    # trainer
    model_name = config['MODEL']['model_name']
    trainer = eval(model_name + 'Trainer')(config=config)
    if not config['LOG']:
        with HiddenPrints():
            metric_value = trainer.train(trial=trial)
    else:
        metric_value = trainer.train(trial=trial)
    return metric_value


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)
    run(config)
    # model_name = ['LightGCN', 'FusionLightGCN']
    # for n in model_name:
    #     config_pth = 'Ciao' + n + '.ini'
    #     for i in range(1):
    #         run(config_pth, True)

