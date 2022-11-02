#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   MutualRecMetric.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/30 16:22   zxx      1.0         None
"""

# import lib
import torch
from .BaseMetric import BaseMetric

class MutualRecMetric(BaseMetric):
    def __init__(self, ks, task, metric_name):
        super(MutualRecMetric, self).__init__(ks, task, metric_name)


