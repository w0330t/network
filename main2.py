import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time

class Classifier(nn.Module):

    def __init__(self) -> None:
        # 初始化父类
        super().__init__()

        # 定义模型
        self.model = nn.Sequential(
            nn.Linear(784, 200),  # 输入层到隐藏层
            # nn.Sigmoid(),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 10),   # 隐藏层到输出层
            # nn.LeakyReLU(0.02),
            nn.Sigmoid()
        )

        # 定义损失函数
        # self.loss_function = nn.MSELoss() #均方误差
        self.loss_function = nn.BCELoss() #二元交叉熵

        # 定义优化器
        # self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01) #随机梯度下降
        self.optimiser = torch.optim.Adam(self.parameters()) 

        # 记录训练进展的计数器和列表
        self.counter = 0
        self.progress = []


    def forward(self, inputs):
        # 前向传播
        return self.model(inputs)


    def train(self, inputs, targets):
        # 计算网络的输出值
        outputs = self.forward(inputs)
        # 计算损失值
        loss = self.loss_function(outputs, targets)

        # 梯度归零，反向传播，并更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # 每隔10个训练样本增加一次计数器的值并保存
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        if self.counter == 0:
            print(f'counter: {self.counter}')


    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(
            ylim=(0, 1.0), 
            figsize=(21, 9), 
            alpha=0.5, 
            marker='.',
            grid=True,
            yticks=(0, 0.25, 0.5))
        plt.show()
        


class MnistDataset(Dataset):

    def __init__(self, file) -> None:
        self.data = np.load(file, allow_pickle=True)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        # 目标图像(label)
        label = self.data[index][1]
        target = torch.zeros((10))
        target[label] = 1.0

        # 图像数据
        image_values = torch.FloatTensor(self.data[index][0])

        return label, image_values, target

    def plot_image(self, index):
        arr = self.data[index][0].reshape(28,28)
        plt.title(f"label: {str(self.data[index][1])}")
        plt.imshow(arr, interpolation='none', cmap='Blues')
        plt.show()


mnist_dataset = MnistDataset('data/training_data.npy')
# 测试代码
# mnist_dataset.plot_image(10)
# print(mnist_dataset[100].shape)

C = Classifier()

epochs = 3
# 循环三遍，开始整活
for i in range(epochs):
    print(f'training epoch {i+1} of {epochs}')
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.train(image_data_tensor, target_tensor)
    print(f'runing: {time.perf_counter()}')

# 绘制分类器损失值
C.plot_progress()

# 测试
mnist_test_dataset = MnistDataset('data/test_data.npy')

# # 随便找一个图
# record = 43
# # 显示record标号的图
# mnist_test_dataset.plot_image(record)
# image_data = mnist_test_dataset[record][1]
# # 调用训练后的神经网络
# output = C.forward(image_data)
# # 绘制输出
# pd.DataFrame(output.detach().numpy()).plot(kind='bar', legend=False, ylim=(0,1))
# plt.show()


# 评分
score = 0
items = 0
for label, image_data_tensor, target_tensor in mnist_test_dataset:
    answer = C.forward(image_data_tensor).detach().numpy()    
    if (answer.argmax() == label):        
        score += 1
    items += 1
print(f'{score}/{items}, {score /items * 100}%')