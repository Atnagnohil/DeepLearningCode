# _*_ coding: utf-8 _*_
'''
时间:      2025/8/6 15:46
@author:  andinm
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.optim as optim

def f(x, y):
    return x*x + 2*y*y

# 生出数据
def creat_dataloader(num_samples):
    x = torch.randn(num_samples)
    y = torch.randn(num_samples)
    z = f(x, y) + torch.randn(num_samples)  # 生成均值为0，方差为1的扰动项
    # 将数据变成1000*3的格式
    dataset = torch.stack([x, y, z], dim=1)
    # print(dataset[0])
    # print(dataset.shape)
    train_size  = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(TensorDataset(train_dataset.dataset.narrow(1, 0, 2),
                                            train_dataset.dataset.narrow(1, 2, 1)),
                              batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_dataset.dataset.narrow(1, 0, 2),
                                            test_dataset.dataset.narrow(1, 2, 1)),
                              batch_size=32, shuffle=False)
    return train_loader, test_loader

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.hidden(x))
        return self.output(out)
# 开始训练
def train(train_loader, test_loader, model, num_epochs, learning_rate):
    # 损失函数
    loss_func = nn.MSELoss()
    # 训练有无学习率调节器的模型
    for with_scheduler in [False, True]:
        train_losses = []
        test_losses = []
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # 指数衰减学习率调节器
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        for epoch in range(num_epochs):
            train_loss = 0
            model.train()
            for i, (x, y) in enumerate(train_loader):
                out = model(x)
                loss = loss_func(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x)
                    loss = loss_func(out, y)
                    test_loss += loss.item()
                test_loss /= len(test_loader)
                test_losses.append(test_loss)
        # 考虑是否更新学习率
        if with_scheduler:
            scheduler.step()

        plot(train_losses, test_losses, num_epochs, with_scheduler)
        # print(list(range(num_epochs)))

# 绘图
def plot(train_losses, test_losses, num_epochs, with_scheduler):
    plt.plot(list(range(num_epochs)), train_losses, label='train')
    plt.plot(list(range(num_epochs)), test_losses, label='test')
    plt.title(f"{'with_out_scheduler' if with_scheduler else 'with_scheduler'}")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    num_samples = 1000
    train_loader, test_loader = creat_dataloader(num_samples)
    '''for i, (x, y) in enumerate(train_loader):
        print(x.shape, y.shape)
        # torch.Size([32, 2]) torch.Size([32, 1])
        break'''
    model = Model()
    num_epochs = 100
    learning_rate = 0.1
    train(train_loader, test_loader, model, num_epochs, learning_rate)



