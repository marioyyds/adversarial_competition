# 对抗挑战赛

[TOC]

## 1. 比赛要求

### 1.1 初赛

初赛的赛制是参赛选手和平台进行对抗，**不限制对抗样本攻击手段（加补丁、噪音等）**，针对海、陆、空等多目标识别的算法模型进行攻击。
![Alt text](imgs/phase1.png)

### 1.2 复赛

复赛中除了使用初赛中的普通可见光数据集做分类外，还增加了遥感图像数据集做**目标检测。**

### 1.3 决赛

**线下比赛**，搭建物理沙盘，摆放高度逼真的军用车辆和基建模型，选手采用攻防对抗的方式进行比赛。使用在模型上**粘贴对抗样本补丁**的方式实现攻击，欺骗防御方的智能识别算法，躲避智能装备的打击。

## 2. 指标

![Alt text](imgs/criterion1.png)  
![Alt text](imgs/criterion2.png)

## 3. 提交要求

![Alt text](imgs/commit.png)

## 4. 安排

**初赛：**

* 合适的base或者弄一个基础工程
  * [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
  * [https://github.com/open-mmlab/mmpretrain](https://github.com/open-mmlab/mmpretrain)
  * [https://github.com/pytorch/vision](https://github.com/pytorch/vision)
  * [https://github.com/jhayes14/adversarial-patch/tree/master](https://github.com/jhayes14/adversarial-patch/tree/master)
* 训练分类模型
* 攻击方法（patch、noise等）选择

思考：

* 攻击的模型是否本身就具有对抗性？
* 这儿的攻击是黑盒攻击，是否可参考之前workshop，使用ensemble？
