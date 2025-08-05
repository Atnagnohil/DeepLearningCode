# _*_ coding: utf-8 _*_
'''
时间:      2025/8/4 23:20
@author:  andinm
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import dataset
from tqdm import *

# 生成数据
def creat_data(num_samples):
    np.random.seed(42)
    x = np.linspace(-5, 5, num_samples)
    '''生成一个均值为 0、标准差为 0.01 的 正态分布噪声；一共生成 num_samples 个随机数，对每个样本加上微小扰动。'''
    y = 0.3*(x**2) + np.random.normal(0, 0.01, num_samples)

    # 转化为Tensor
    x = torch.unsqueeze(torch.from_numpy(x), dim=1).float()
    y = torch.unsqueeze(torch.from_numpy(y), dim=1).float()

    # 封装成数据包格式
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset



# 定义模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.output(out)
        return out

# 训练模型
def train_model(model, dataset ,learning_rate, num_epochs, batch_size, momentum, names):
    losses = [[] for _ in range(len(names))]
    for i in range(3):
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum[i])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size[i], shuffle=True)
        for epoch in tqdm(range(num_epochs), desc=names[i], leave=True, unit='epoch'):
            '''| 组件                  | 作用                                                        |
| ------------------- | --------------------------------------------------------- |
| `range(num_epochs)` | 表示训练要进行的轮数（epoch 次数）                                      |
| `tqdm(...)`         | 将 `range(...)` 包装成一个带有实时进度条的可迭代对象                         |
| `desc=names[i]`     | 进度条左侧的描述文字，例如：`Batch`、`Stochastic`、`Mini_Batch`           |
| `leave=True`        | 训练完成后是否保留进度条（`True` 保留，不会被清屏）                             |
| `unit='epoch'`      | 单位名，显示在进度条右侧（如：`1000/1000 [00:01<00:00, 780.15 epoch/s]`） |
'''
            x, y = next(iter(data_loader))  # 每次下降选择一个batch_size的标注
            out = model(x)
            loss = loss_fn(out, y)
            # 梯度请零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[i].append(loss.item())
    return losses

# 绘制损失变化
def show(losses, num_epochs, names):
    for idx, loss_li in enumerate(losses):
        plt.figure(figsize=(12, 4))
        plt.plot(range(num_epochs), loss_li), plt.xlabel('epoch'), plt.ylabel('loss')
        plt.title(names[idx])
        plt.show()


def main():
    # 定义超参数
    NUM_SAMPLES = 1000
    INPUT_SIZE = 1
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1000

    # 创建数据
    dataset = creat_data(NUM_SAMPLES)

    names = ["Batch", "Stochastic", "Mini_Batch"]
    ''' 小批量梯度下降最好为2的幂次（64， 128， 256），计算效率高'''
    batch_size = [NUM_SAMPLES, 1, 128]
    '''Batch 和 Mini-Batch：
        都是“平均梯度”，更新较稳定。
        通常可以启用动量，帮助加速下降并跳出鞍点。'''
    momentum = [1, 0, 1]
    '''
    momentum=1
    这就变成了累加所有梯度，相当于永远记住过去的梯度，不会有任何衰减 —— 容易造成震荡或发散。
    最长设置为0.9
    '''
    model = Model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    losses = train_model(model, dataset, LEARNING_RATE, NUM_EPOCHS, batch_size, momentum, names)
    # print(losses[1])
    show(losses, NUM_EPOCHS, names)

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"device: {device}")
    main()
'''Batch: 100%|██████████| 1000/1000 [00:06<00:00, 153.98epoch/s]
Stochastic: 100%|██████████| 1000/1000 [00:00<00:00, 3211.85epoch/s]
Mini_Batch: 100%|██████████| 1000/1000 [00:01<00:00, 833.91epoch/s]
'''