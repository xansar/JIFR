#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LightGCNTrainer.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/11/4 21:19   zxx      1.0         None
"""

# import lib

import dgl
import torch
import numpy as np
from tqdm import tqdm, trange
import os

from .BaseTrainer import BaseTrainer

class LightGCNTrainer(BaseTrainer):
    def __init__(self, config):
        super(LightGCNTrainer, self).__init__(config)

    def train(self):
        loss_name = ['Loss']
        self._train(loss_name)
