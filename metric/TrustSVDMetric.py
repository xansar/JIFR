#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   TrustSVDMetric.py
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/25 11:38   zxx      1.0         None
"""


# import lib
import numpy as np
import torch

from .BaseMetric import BaseMetric

class TrustSVDMetric(BaseMetric):
    def __init__(self, ks, task, metric_name):
        super(TrustSVDMetric, self).__init__(ks, task, metric_name)