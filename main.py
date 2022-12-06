from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


def create_wights(row, column) -> torch.Tensor:
    """创建权重

    Args:
        count (int): 矩阵的行数量
        column (int): 矩阵的列数量

    Returns:
        torch.Tensor: 返回指定权重数量的随机数
    """
    return torch.normal(0, 0.01, size=(row, column), requires_grad=True)


def create_biases(count) -> torch.Tensor:
    """创建偏置

    Args:
        count (int): 创建的数量

    Returns:
        torch.Tensor: 返回指定权重数量的随机数
    """
    return torch.zeros(count, 1, requires_grad=True)


def network(expamles, weights, biases) -> torch.Tensor:
    """神经网络模型

    Args:
        expamles (_type_): 样本
        weights (_type_): 权重
        biases (_type_): 偏置

    Returns:
        torch.Tensor: 实际计算的结果
    """

    expamles = torch.Tensor(expamles).reshape(len(expamles), 1) \
        if type(expamles) == np.ndarray \
        else expamles.reshape(len(expamles), 1)
    expamles.requires_grad = True

    r = expamles + 1e-9
    for i in range(len(weights)):
        r = torch.relu(torch.matmul(weights[i], r) + biases[i])

    return r


def cost(y:int, y_hat:torch.Tensor) -> torch.Tensor:
    """代价函数，计算损失用的。用真实的值和模型训练出来的值做对比。
        除2是为了方便计算

    Args:
        y (int): 正确的值。识别数字的int类型，需要格式化为tensor后做计算
        y_hat (torch.Tensor): 由网络训练出来的值。

    Returns:
        torch.Tensor: 平均代价，用来评估当前模型训练的成绩。
    """
    t = torch.zeros(y_hat.shape)
    t[y] = 1
    return (y_hat - t).pow(2).sum() / 2


def gd(param:torch.Tensor, lr:float):
    with torch.no_grad():
        param = param - lr * param.grad
        param.requires_grad = True
        return param


expamles = np.load('data/training_data.npy', allow_pickle=True)

weights = [create_wights(16, 784), create_wights(16, 16), create_wights(10, 16)]
biases = [create_biases(16), create_biases(16), create_biases(10)]
lr = 0.03

for i in range(len(expamles)):
    model_result = network(expamles[i][0], weights, biases)
    cost_result = cost(expamles[i][1], model_result)
    cost_result.backward()
    for j in range(len(weights)):
        weights[j] = gd(weights[j], lr)
        biases[j] = gd(biases[j], lr)
    if (i + 1) % 5000 == 0:
        print(f'index: {i + 1}, Cost: {cost_result}')


test_expamles = np.load('data/test_data.npy', allow_pickle=True)

success = 0

for i in range(len(test_expamles)):
    test_result = network(test_expamles[i][0], weights, biases)
    test_result_index = torch.nonzero(test_result == test_result.max()).squeeze()[0]

    if test_result.sum() != 0 and \
        torch.nonzero(test_result == test_result.max()).squeeze()[0] == test_expamles[i][1]:
        success += 1

print(f'Success: {success/len(test_expamles) * 100}%')