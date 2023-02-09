#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   logger.py    
@Contact :   xansar@ruc.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/12/9 15:40   zxx      1.0         None
"""

# import lib
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 20:51
# @Author  : xx
import logging
import os
import time


def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join('.',
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger

# 获取logger
logger = getLogger()
# logger.info('train|epoch:{epoch}\tstep:{step}/{all_step}\tloss:{loss:.4f}'.format(epoch=epo, step=index + 1,all_step=len(train_loader), loss=loss.item()))  # 打印训练日志
epoch = 14
step = 233
all_step = 2
loss = 0.6931
for e in range(128):
    for s in range(2):
        logger.info(f'train|epoch:{e}\tstep:{s}/{all_step}\tloss:{loss:.4f}')