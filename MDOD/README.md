# MDOD

## 目录结构
```
  ├── dataset: 数据集
  │    ├── DUTS-TE: 测试集
  │    │     ├── DUTS-TE-Image: 原始图片
  │    │     └── DUTS-TE-Mask: 标注图片
  │    └── DUTS-TR: 训练集
  │          ├── DUTS-TR-Image: 原始图片
  │          └── DUTS-TR-Image: 标注图片
  ├── pre-dataset: 数据集预处理
  ├── inference: 推理模型
  ├── results: 训练结果
  ├── runs: 训练可视化文件
  ├── utils: 工具
  │     ├── split_data.py: 分割数据
  │     ├── to_gray.py: 转灰度
  │     └── to_mask.py: 转蒙版
  ├── train_utils: 训练验证网络时使用到的工具
  ├── save_weights: 相关训练权重
  ├── src: 源码
  │     └── model.py: 模型搭建文件
  ├── train.py: 针对单GPU或者CPU的用户使用
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  └── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
```

## 执行指令
安装tensorboard可视化界面
```
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```
```
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装opencv-python
```
pip install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```
```
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## 开始训练
训练脚本
```
nohup python train.py &
```
数据可视化
```
nohup tensorboard --logdir=runs &
```
# 网络结构
![U2NETPRmodel.png](intro%2FU2NETPRmodel.png)
