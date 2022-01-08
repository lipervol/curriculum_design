#! coding:UTF-8
#@Liupengbin https://github.com/lipervol
import sys
import numpy as np
import scipy.io as sio
import os
import torch
from torch.utils.data import DataLoader
import spectral as spy
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import train
from train import ResnetBlock, ResNet

# 载入图像和标签
image_path = "./data"  # 数据路径
data = sio.loadmat(os.path.join(image_path, "Indian_pines_corrected.mat"))["indian_pines_corrected"]
gt = sio.loadmat(os.path.join(image_path, "Indian_pines_gt.mat"))["indian_pines_gt"]

# 降维并分割图像
data = train.svd_decomposition(data, 64)
data, labels = train.create_dataset(data, gt, 8, False)

# 加载模型
model_path = "./model_save.pth"  # 模型路径
if os.path.exists(model_path):
    model = torch.load(model_path)
    print("[*]加载模型... ...")
else:
    print("[*]未找到模型... ...")
    sys.exit(0)

# 准备数据集
dataset = train.MyDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 预测
gt_pred = np.uint8(np.zeros((gt.shape[0] * gt.shape[1])))
device = torch.device("cuda")  # 使用GPU
model = model.to(device)
idx = 0
print("[*]开始预测... ...")
model.eval()
with torch.no_grad():
    for datas in dataloader:
        img, label = datas
        if label == 0:
            idx += 1
            pass
        else:
            img = img.type(torch.float32)
            img = img.to(device)
            pred = model(img)
            pred = pred.argmax(dim=1)
            gt_pred[idx] = pred.item() + 1
            idx += 1
        print("\r[*]已完成：%.2f" % (float(idx / (gt.shape[0] * gt.shape[1])) * 100), "%", end='')
gt_pred = np.uint8(np.reshape(gt_pred, (gt.shape[0], gt.shape[1])))

# 精度分析
kappa = cohen_kappa_score(gt.reshape(-1, 1), gt_pred.reshape(-1, 1))  # kappa系数
print("\nKappa系数为：", kappa)
matrix = confusion_matrix(gt[gt > 0].reshape(-1, 1), gt_pred[gt > 0].reshape(-1, 1),
                          labels=[i + 1 for i in range(16)])  # 混淆矩阵
print("混淆矩阵为：\n", matrix)
sns.set()
f, ax = plt.subplots()
sns.heatmap(matrix, annot=True, ax=ax)  # 画热力图
ax.set_title('confusion matrix')
ax.set_xlabel('predict')
ax.set_ylabel('true')

# 显示结果
print("\n[*]显示结果... ...")
gt_view = spy.imshow(classes=gt)
gt_pred_view = spy.imshow(classes=gt_pred)
plt.show()
plt.pause(300)
