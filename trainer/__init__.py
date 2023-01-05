#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/25 10:01   zxx      1.0         None
"""
from .utils import *

# without social
from .MFTrainer import *
from .LightGCNTrainer import *
from .NCLTrainer import *
from .SSCLTrainer import *

# classical social
from .TrustSVDTrainer import *
from .SVDPPTrainer import *
from .SorecTrainer import *
from .SocialMFTrainer import *
from .NJBPTrainer import *

# gnn social
from .FusionLightGCNTrainer import *
from .DiffnetPPTrainer import *
from .MutualRecTrainer import *
from .GraphRecTrainer import *
from .SocialLGNTrainer import *

# link
from .AATrainer import *
from .Node2VecTrainer import *
