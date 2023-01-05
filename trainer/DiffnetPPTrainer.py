#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   DiffnetPPTrainer.py
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

class DiffnetPPTrainer(BaseTrainer):
    def __init__(self, config, trial=None):
        super(DiffnetPPTrainer, self).__init__(config, trial=trial)
