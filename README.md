# 代码

## 0 前言

围绕氮化硅轴承球微裂纹深度细节特征的边缘梯度模糊、轮廓样条畸形、类别多样重叠的繁杂现象，结合图像处理与深度学习融合多尺度参量特征完整提取方法的难点。提出多元化图像处理、多维度目标检测、多方位语义分割的耦合新方法，旨在实现多尺度参量氮化硅轴承球微裂纹深度细节特征完整提取，为助力氮化硅轴承在极端工况环境全面推广夯实理论基础

## 1 文件结构

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

## 2 简介

### 2.1 基于多元化图像处理的单尺度参量氮化硅轴承球微裂纹深度细节特征研究

> 围绕氮化硅轴承球微裂纹深度细节特征存在边缘梯度模糊、轮廓样条畸形、类别多样重叠的无损识别与完整提取痛点。同时，剖析氮化硅轴承球微裂纹深度细节特征的材料属性、信息元素、检测方法等多学科融合理论。结合中值滤波、拉普拉斯算子、灰度阈值分割、区域分割、边缘检测分割、特定识别区域分割等图像处理方法。从单尺度参量范畴设计氮化硅轴承球微裂纹深度细节特征的多元化图像处理方法，实现单图像单目标参量氮化硅轴承球微裂纹深度细节特征的无损识别与完整提取

<details><summary>变尺度截断中值滤波局部性增强</summary>
首先，单图像单目标参量微裂纹深度细节特征图像的信号全覆盖半径边界扩展过程，保证边缘梯度模糊、轮廓样条畸形、类别多样重叠均在信号范围内，防止数据缺失产生边界效应。其次，信号全覆盖表征图像局部曲率拟合，通过二阶导数核估算每个像素包含区域的曲率，实现每个像素点包含区域的轮廓量化。再次，轮廓量化后表征图像圆形窗口自适应处理，优化Sigmoid函数进行曲率标准化，根据曲率动态调整每个像素滤波窗口获取局部窗口。最后，局部窗口表征图像阈值截断剔除，通过剔除局部窗口非零元素，比较当前像素与中值之间的阈值差异，结合阈值界限实现消噪与平滑图像细节的自适应平衡，实现单图像单目标参量微裂纹深度细节特征图像的局部性增强效果。
</details>
<details><summary>检测驱动活动轮廓全局性分割</summary>
率先，单图像单目标参量微裂纹深度细节特征图像边缘信号连通带宽边缘识别，通过连通组件分析以扫描二值化图像，并对每个独立连通区域进行标记，利用不同颜色线条区分各个组件。然后，遍历每个搜寻连通组件，比对连通区域与既定面积阈值，识别并保留符合条件的标定区域以更新布尔索引，同时剔除噪声和无关信号，直到遍历结束更新BWf以作为下一步迭代初始ROI，实现单图像单目标参量微裂纹深度细节特征图像轮廓线分割。再后，构建能量函数指导轮廓逐渐趋于平滑，在迭代过程轮廓位置按能量最小化原则不断优化调整，直至变化量降至设定阈值以下，形成紧贴目标边缘轮廓，并将其作为单图像单目标参量氮化硅轴承球微裂纹深度细节特征图像分割的结果输出。最后，通过形态学膨胀操作补充轮廓内部的小裂隙，并连接邻近的裂痕片断，确保整体单图像单目标参量氮化硅轴承球微裂纹深度细节特征图像连贯性。
</details>

### 2.2 基于多维度目标检测的双尺度参量氮化硅轴承球微裂纹深度细节特征研究

> 针对单图像单目标参量氮化硅轴承球微裂纹深度细节特征多元化图像处理存在灰度值自身局部界定问题；同时，多元化图像处理在识别与提取特征图像层面只能实现单一图像参量的处理。为此，结合多尺度CNN特征、深度神经网络、显著性模型、空间一致性模型、多层次图像分割、方向梯度直方图、注意力机制等深度学习目标检测方法。从双尺度参量范畴设计氮化硅轴承球微裂纹深度细节特征的多维度目标检测方法，实现多图像单目标参量氮化硅轴承球微裂纹深度细节特征的无损识别与完整提取。

<details><summary>混合交叉注意力模块自适应重组</summary>
起初，多图像单目标参量微裂纹深度细节特征图像局部放大及色彩映射，同时借助与非运算获取概率预测图与真实标注特征之间的误差，分析微裂纹深度细节特征类膨胀现象。另外，多尺度特征块嵌入向量融合，利用多尺度特征块嵌入层将不同尺度的卷积核解码为嵌入向量阵列，通过多尺度嵌入向量融合模块聚集，以实现特征图像在相同特征级别目标定位和高语义特征的融合。再者，构建混合交叉注意力模块，对嵌入向量进行归一化处理均衡数据差异，串联标记所有嵌入向量沿通道创建相关的键值，通过Softmax函数对矩阵值加权将深度卷积投影于交叉注意力模块，实现传递信息至DCA模块顺序输出。最后，神经网络推理预测结果与真实标注之间的差异性，通过与非运算计算得到预测结果与真实标注的特征误差，对微裂纹深度细节特征主体部分局部放大，判断多图像单目标参量微裂纹深度细节特征的类膨胀现象改善情况。
</details>
<details><summary>多尺度方向梯度直方图系统性预测</summary>
最初，多图像单目标参量微裂纹深度细节特征图像嵌入基准网络(U2-Net)进行自适应推理，借助叠加合成方式将预测结果与真实标注特征对比分析，实现微裂纹深度细节特征整体初步识别。后者，为直观展示微裂纹深度细节特征缺失现象，选取差异性灰度阈值对原始特征图像进行灰度映射，剖析微裂纹深度细节特征缺失规律。再后，构建多尺度方向梯度直方图，利用伽马矫正对输入特征图像非线性细节增强，将特征划分为多个固定窗口并计算其X/Y方向的方向梯度，统计方向梯度叠加数值绘制方向梯度直方图。最后，微裂纹深度细节特征神经网络预测，将特征描述符几何中心相连预测轮廓并作为辅助任务伪标签，经过基准网络(+HOG)进行二次训练实现结果预测。
</details>

### 2.3 基于全方位语义分割的多尺度参量氮化硅轴承球微裂纹深度细节特征研究

> 综合多图像单目标参量氮化硅轴承球微裂纹深度细节特征多维度目标检测方法存在提取特征目标单一的问题，无法实现单一图像多目标参量特征的分割。因此，结合训练策略、损失函数、网络结构、上下采样、多尺度特征融合、多类型注意力机制等深度学习语义分割方法。从多尺度参量范畴设计氮化硅轴承球微裂纹深度细节特征的全方位语义分割方法，实现多图像多目标参量氮化硅轴承球微裂纹深度细节特征的无损识别与完整提取。

<details><summary>边缘通道注意力机制完整性识别</summary>
开始，多图像多目标参量微裂纹深度细节特征数据集特征分析，沿斜向、水平和垂直剖面的数值梯度变化对比特征差异性，解析点-线-面微裂纹深度细节特征的显著性与完整性语义信息。第二，提取点-线-面微裂纹深度细节特征数据集特征易混淆的语义信息，剖析特征图像的背景与目标特征之间的相似语义信息的内在差别，为设计时效性、准确性、融合性的边缘通道注意力机制奠定前期基础。其三，为了实现特征图像语义信息完整、清晰还原微裂纹深度细节特征信号，对首次提取的微裂纹深度细节特征上采样全局信息与未采样细节信息输入双卷积模块，分别获得特征图像的深层信息和浅层信息。最后，将浅层信息下采样为细节信息，再将细节信息与深层信息融合，通过加权计算获取注意力增强的特征信息，从而实现多图像多目标参量微裂纹深度细节特征清晰和饱满的语义信息。
</details>
<details><summary>加权门控注意力机制高效性提取</summary>
起首，多图像多目标参量微裂纹深度细节特征残差结构特征图像深层信息提取，通过上采样降低通道数且利用维度不变卷积特征提取，激活函数调整特征信息增强特征的表达能力。接着，为了在同层融合保留充分的特征图像语义信息，将深层信息与浅层信息同时输入模型初步融合，利用加权方法与浅层信息融合，强化同层内的特征图像信息。再三，比较ReLU、Sigmoid和H-Swish三种激活函数的数据曲线，通过不同激活函数对数据整形作用，调整特征图像的阈值范围实现特征信号区分。最后，同层下采样信息与合成信息融合输出权重，与同层下采样信息相结合，提高输出信息的信息丰度，实现同层内多图像多目标参量微裂纹深度细节特征的高效提取。
</details>

## 3 使用

### 3.1 安装依赖

安装tensorboard可视化界面
```
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装opencv-python
```
pip install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 3.2 开始训练

训练脚本
```
nohup python train.py &
```
数据可视化
```
nohup tensorboard --logdir=runs &
```

## 4 实验结果
### 4.1 DIP

| 组别 | 算法  类型 | 圆窗 | 截断  | 轮廓  | PSNR/dB     | EPI        | 精确率/%  | 准确率/%  | 召回率/%  |
| ---- | ---------- | ---- | ----- | ----- | ----------- | ---------- | --------- | --------- | --------- |
| ①    | MF         | -    | -     | **√** | -           | -          | 88.56     | 88.97     | 86.76     |
|      |            | -    | **√** | **√** | 38.4437     | 0.3514     | 87.36     | 92.38     | 89.39     |
| ②    | BF         | -    | -     | √     | 38.0650     | 0.1824     | 89.26     | 89.83     | 87.32     |
| ③    | SVM        | -    | -     | -     | -           | -          | 90.01     | 91.83     | 89.74     |
|      |            | √    | √     | -     | **40.2639** | **0.4533** | 92.53     | 94.96     | 90.01     |
| ④    | RG         | -    | -     | -     | -           | -          | 88.69     | 91.28     | 87.70     |
|      |            | √    | √     | -     | **40.2639** | **0.4533** | 91.59     | 91.46     | 89.62     |
| ⑤    | 本文  算法 | √    | -     | √     | 37.6475     | 0.3752     | 91.25     | 93.61     | 88.46     |
|      |            | -    | √     | √     | 39.9394     | 0.3640     | 92.79     | 92.37     | 89.25     |
|      |            | √    | √     | √     | **40.2639** | **0.4533** | **94.33** | **95.45** | **91.69** |

### 4.2 MDOD

| **网络模型** | **Params(M)** | **GFLOPs** | **mIoU**(%) | **Fφ(%)**  | **Sφ(%)**  | **Eφ(%)**  | **AUC（%)**  | **M（10-3)** |
| ------------ | ------------- | ---------- | ----------- | ---------- | ---------- | ---------- | ------------ | ------------ |
|              | **(0~∞)↓**    | **(0~∞)↓** | **[0~1]↑**  | **[0~1]↑** | **(0~1)↑** | **(0~1)↑** | **(0.5~1)↑** | **(0~∞)↓**   |
| PoolNet      | 278.5         | 38.2       | 60.56       | 78.23      | 73.62      | 84.19      | 77.84        | 31           |
| Res2Net      | 45.2          | **8.1**    | 64.63       | 83.00      | 82.62      | 90.92      | 84.28        | 13           |
| BASNet       | 87.1          | 97.7       | 76.28       | 89.34      | 90.82      | 97.05      | **98.77**    | 10           |
| MINet        | 162.3         | 42.7       | 79.59       | 90.03      | 90.82      | 97.63      | 98.29        | 9            |
| MCCS         | 43.9          | 28.8       | **89.49**   | **94.65**  | **94.75**  | **98.95**  | 96.69        | **4**        |
| MCCS-Lite    | **1.1**       | 9.7        | 83.36       | 91.20      | 91.34      | 97.76      | 94.76        | 7            |

### 4.3 ODSS

| Methods          | Backbone  | mIoU′ | Mpa′ | Precision′ | Recall′ |
| ---------------- | --------- | ------ | ------ | ---------- | ------- |
| Unet             | Resnet101 | 81.04  | 92.34  | 87.23      | 90.97   |
| Unet             | Vgg16     | 81.29  | 91.57  | 87.36      | 91.26   |
| U-net+           | Vgg16     | 83.26  | 92.16  | 90.87      | 92.54   |
| ECEM_Unet        | Vgg16     | 82.23  | 92.34  | 89.47      | 92.2    |
| WAGM_Unet        | Vgg16     | 82.62  | 92.44  | 88.56      | 91.48   |
| ECEM_WAGM_U-net+ | Vgg16     | 85.31  | 94.2   | 92.26      | 94.8    |
