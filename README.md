##  北化课程设计（冉琼）

### 任务简介

高光谱图像降维和分类，不可使用PCA和k近邻方法

数据来源：http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes 

### 方法介绍

采用奇异值分解算法(SVD)进行降维(154,154,200)-->(154,154,64)

图像切割为为(154\*154,64,8,8)后，采用以下ResNet进行分类：

![ResNet](https://github.com/lipervol/curriculum_design/blob/master/ResNet.png)

### 实现效果

随机挑选40%作为训练集，60%作为测试集，迭代48次

左边是ground truth，右边为预测输出

![result](https://github.com/lipervol/curriculum_design/blob/master/rst.png)

Kappa系数为：0.9977835341413226

奇异值矩阵：

![Hot](https://github.com/lipervol/curriculum_design/blob/master/confusion_matrix.png)