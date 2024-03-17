# 代码
## 0前言

## 1文件结构

```
  ├── DIP:  多元化图像处理
      ├── detection_drives.m
      ├── truncated_filter.m
  ├── MDOD: 多维度目标检测
    ├── dataset: 数据集
  	│    ├── DUTS-TE: 测试集
  	│    │     ├── DUTS-TE-Image: 原始图片
  	│    │     └── DUTS-TE-Mask: 标注图片
  	│    └── DUTS-TR: 训练集
 	│          ├── DUTS-TR-Image: 原始图片
 	│          └── DUTS-TR-Image: 标注图片
  	├── inference: 推理模型
  	├── utils: 工具
  	│     ├── split_data.py: 分割数据
  	│     ├── to_gray.py: 转灰度
  	│     └── to_mask.py: 转蒙版
 	├── train_utils: 训练验证网络时使用到的工具
  	├── src: 源码
  	│     └── model.py: 网络模型
  	├── train.py: 针对单GPU或者CPU的用户使用
  	├── train_multi_GPU.py: 针对使用多GPU的用户使用
  	└── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  └── ODSS: 全方位语义分割
    ├── nets: 网络结构
  	├── utils: 工具
  	│     ├── dataloader.py: 数据集加载
  	│     ├── utils.py: 处理工具
  	├── train.py: 针对单GPU或者CPU的用户使用
  	└── unet.py: 网络模型
```

## 2简介

### 2.1基于多元化图像处理的单尺度参量氮化硅轴承球微裂纹深度细节特征研究

围绕氮化硅轴承球微裂纹深度细节特征存在边缘梯度模糊、轮廓样条畸形、类别多样重叠的无损识别与完整提取痛点。同时，剖析氮化硅轴承球微裂纹深度细节特征的材料属性、信息元素、检测方法等多学科融合理论。结合中值滤波、拉普拉斯算子、灰度阈值分割、区域分割、边缘检测分割、特定识别区域分割等图像处理方法。从单尺度参量范畴设计氮化硅轴承球微裂纹深度细节特征的多元化图像处理方法，实现单图像单目标参量氮化硅轴承球微裂纹深度细节特征的无损识别与完整提取

### 2.2基于多维度目标检测的双尺度参量氮化硅轴承球微裂纹深度细节特征研究

针对单图像单目标参量氮化硅轴承球微裂纹深度细节特征多元化图像处理存在灰度值自身局部界定问题；同时，多元化图像处理在识别与提取特征图像层面只能实现单一图像参量的处理。为此，结合多尺度CNN特征、深度神经网络、显著性模型、空间一致性模型、多层次图像分割、方向梯度直方图、注意力机制等深度学习目标检测方法。从双尺度参量范畴设计氮化硅轴承球微裂纹深度细节特征的多维度目标检测方法，实现多图像单目标参量氮化硅轴承球微裂纹深度细节特征的无损识别与完整提取。

### 2.3基于全方位语义分割的多尺度参量氮化硅轴承球微裂纹深度细节特征研究

综合多图像单目标参量氮化硅轴承球微裂纹深度细节特征多维度目标检测方法存在提取特征目标单一的问题，无法实现单一图像多目标参量特征的分割。因此，结合训练策略、损失函数、网络结构、上下采样、多尺度特征融合、多类型注意力机制等深度学习语义分割方法。从多尺度参量范畴设计氮化硅轴承球微裂纹深度细节特征的全方位语义分割方法，实现多图像多目标参量氮化硅轴承球微裂纹深度细节特征的无损识别与完整提取。
