# joint_rec
## 快速开始
1. 处理数据：
  首先运行`data/utils`中的`split_data.py`，调整好比例进行切分
2. 调整config文件中的各项参数和属性
3. 运行run文件，修改其中的参数config_pth，也支持命令行修改config位置
4. 训练记录会存放至log文件夹中对应的模型下，命名采用`task_randomseed_modelname.txt`的格式
5. 训练好的模型会保存在save文件夹下的模型文件夹中，命名采用`task_randomseed_modelname.pth`的格式
