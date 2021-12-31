#! coding:UTF-8
import numpy as np
import os
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as op
import torch


# 利用截断SVD降维
def svd_decomposition(input_data, channels):
    shape = input_data.shape
    output_data = np.reshape(input_data, (-1, shape[2]))
    svd = TruncatedSVD(channels)
    output_data = svd.fit_transform(output_data)
    output_data = np.reshape(output_data, (shape[0], shape[1], channels))
    return output_data


# 分割图像
def create_dataset(input_data, input_label, window_size, remove_zeros=True):
    shape = input_data.shape
    margin = int(window_size / 2)
    pad = np.zeros((shape[0] + 2 * margin, shape[1] + 2 * margin, shape[2]))
    pad[margin:shape[0] + margin, margin:shape[1] + margin, :] = input_data
    dataset = np.zeros((shape[0] * shape[1], window_size, window_size, shape[2]))
    label_np = np.zeros((shape[0] * shape[1]))
    idx = 0
    for i in range(margin, pad.shape[0] - margin):
        for j in range(margin, pad.shape[1] - margin):
            split_data = pad[i - margin:i + margin, j - margin:j + margin, :]
            dataset[idx] = split_data
            label_np[idx] = input_label[i - margin, j - margin]
            idx += 1
    if remove_zeros:
        dataset = dataset[label_np > 0, :, :, :]
        label_np = label_np[label_np > 0]
        label_np -= 1
    return dataset, label_np


# 数据集
class MyDataset(Dataset):
    def __init__(self, mydata, mylabels):
        self.mydata = np.reshape(mydata, (mydata.shape[0], mydata.shape[3], mydata.shape[1], mydata.shape[2]))  # 调整形状
        self.mylabels = mylabels

    def __getitem__(self, idx):
        output = (self.mydata[idx], self.mylabels[idx])
        return output

    def __len__(self):
        return len(self.mydata)


# Resnet网络
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, residual_path=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride=strides, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.a1 = nn.ReLU()

        self.c2 = nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_channels)

        if self.residual_path:
            self.down_c1 = nn.Conv2d(in_channels, out_channels, (1, 1), stride=strides, padding=0, bias=False)
            self.down_b1 = nn.BatchNorm2d(out_channels)

        self.a2 = nn.ReLU()

    def forward(self, inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        y = self.b2(x)
        if self.residual_path:
            residual = self.down_c1(residual)
            residual = self.down_b1(residual)
        model_output = self.a2(y + residual)
        return model_output


# ResNet网络
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(128)
        self.a1 = nn.ReLU()
        self.r1 = ResnetBlock(128, 128, residual_path=False)
        self.r2 = ResnetBlock(128, 128, residual_path=False)
        self.r3 = ResnetBlock(128, 256, strides=2, residual_path=True)
        self.r4 = ResnetBlock(256, 256, residual_path=False)
        self.r5 = ResnetBlock(256, 512, strides=2, residual_path=True)
        self.r6 = ResnetBlock(512, 512, residual_path=False)
        self.p1 = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(512, 16)

    def forward(self, inputs):
        model_outputs = self.c1(inputs)
        model_outputs = self.b1(model_outputs)
        model_outputs = self.a1(model_outputs)
        model_outputs = self.r1(model_outputs)
        model_outputs = self.r2(model_outputs)
        model_outputs = self.r3(model_outputs)
        model_outputs = self.r4(model_outputs)
        model_outputs = self.r5(model_outputs)
        model_outputs = self.r6(model_outputs)
        model_outputs = self.p1(model_outputs)
        model_outputs = self.f1(model_outputs)
        model_outputs = self.l1(model_outputs)
        return model_outputs


if __name__ == "__main__":
    data_save_path = "./data_save.npy"  # 数据保存路径
    label_save_path = "./label_save.npy"
    if os.path.exists(data_save_path) and os.path.exists(label_save_path):
        data = np.load(data_save_path)  # 读入数据
        labels = np.load(label_save_path)
        print("[*]读入数据... ...")
    else:
        image_path = "./data"  # 数据路径
        data = sio.loadmat(os.path.join(image_path, "Indian_pines_corrected.mat"))["indian_pines_corrected"]  # 读入数据
        labels = sio.loadmat(os.path.join(image_path, "Indian_pines_gt.mat"))["indian_pines_gt"]

        data = svd_decomposition(data, 64)  # 降维

        data, labels = create_dataset(data, labels, 8)  # 分割

        np.random.seed(8)  # 打乱顺序
        np.random.shuffle(data)
        np.random.seed(8)
        np.random.shuffle(labels)

        np.save(data_save_path, data)  # 保存数据
        np.save(label_save_path, labels)
        print("[*]保存数据... ...")

    div = int(0.2 * len(data))  # 分割测试集和训练集
    train_data = MyDataset(data[0:-div], labels[0:-div])
    test_data = MyDataset(data[-div:], labels[-div:])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  # 创建dataloader
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    save_path = "./model_save.pth"  # 加载模型
    if os.path.exists(save_path):
        model = torch.load(save_path)
        print("[*]加载模型... ...")
    else:
        model = ResNet()

    device = torch.device("cuda")  # 使用GPU训练
    model = model.to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = op.Adam(model.parameters(), lr=1e-3)

    train_epochs = 16  # 迭代次数
    print("[*]开始训练... ...")
    for epochs in range(train_epochs):
        model.train()  # 训练模式
        for datas in train_loader:
            imgs, label = datas
            imgs = imgs.type(torch.float32)
            label = label.type(torch.int64)
            imgs = imgs.to(device)
            label = label.to(device)

            outputs = model(imgs)
            losses = loss(outputs, label)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print("[*]次数:", epochs + 1, "\n训练集误差:", losses.item())

        model.eval()  # 测试模式
        with torch.no_grad():
            total_correct = 0
            total_num = 0

            for datas in test_loader:
                imgs, label = datas
                imgs = imgs.type(torch.float32)
                label = label.type(torch.int64)
                imgs = imgs.to(device)
                label = label.to(device)

                outputs = model(imgs)
                pred = outputs.argmax(dim=1)

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += imgs.size(0)
        acc = total_correct / total_num
        print("测试集正确率:", acc)

    print("[*]训练结束... ...")
    torch.save(model, save_path)
    print("[*]保存模型... ...")
