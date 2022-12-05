#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MFTrainer.py
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

class MFTrainer(BaseTrainer):
    def __init__(self, config):
        super(MFTrainer, self).__init__(config)
