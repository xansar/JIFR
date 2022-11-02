#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LightGCNMetric.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/26 16:28   zxx      1.0         None
"""


# import lib
import torch
from .BaseMetric import BaseMetric

class LightGCNMetric(BaseMetric):
    def __init__(self, ks, task, metric_name):
        super(LightGCNMetric, self).__init__(ks, task, metric_name)
