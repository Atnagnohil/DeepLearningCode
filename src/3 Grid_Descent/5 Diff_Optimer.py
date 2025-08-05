# _*_ coding: utf-8 _*_
'''
时间:      2025/8/5 22:04
@author:  andinm
'''
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim


def f(x, y):
    return x ** 2 + 2 *y*y

def process_date(X, Y, Z):
    '''
    当前数据及没有将标签和特征切分，需要切分
    '''
    dataset = torch.stack([X, Y, Z], dim=1)  # 安装列的方向整合
    # print(dataset[0])

    '''数据集创建：torch.stack([X, Y, Z], dim=1) 将 X, Y, Z 按列拼接成形状为 (n_samples, 3) 的张量，前两列是特征，第3列是标签。
    数据集分割：random_split 按 8:2 比例将数据集分为训练集和测试集，train_set 和 test_set 是 Subset 对象。'''
    # 按照8:2划分数据集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(TensorDataset(train_set.dataset.narrow(1, 0, 2),  # 特征：第0-1列
                             train_set.dataset.narrow(1, 2, 1)),  # 标签：第2列
                             batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_set.dataset.narrow(1, 0, 2),  # 特征：第0-1列
                             test_set.dataset.narrow(1, 2, 1)),  # 标签：第2列
                             batch_size=32, shuffle=False)
    return train_loader, test_loader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.hidden(x))
        return self.output(out)

def train(num_epochs, train_loader, test_loader, models, opts, loss_fn, num_opt):
    train_losses_his, test_losses_his = [[] for _ in range(num_opt)], [[] for _ in range(num_opt)]
    for epoch in range(num_epochs):
        # 当前epoch在训练集的总损失列表
        train_losses = [0] * num_opt
        for x, y in train_loader:
            for index, model, optimizer, loss_history in zip(range(num_opt), models, opts, train_losses_his):
                model.train()
                out = model(x)
                loss = loss_fn(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses[index] += loss.item()

        test_losses = [0] * num_opt
        with torch.no_grad():
            for x, y in test_loader:
                for index, model, optimizer, loss_history in zip(range(num_opt), models, opts, train_losses_his):
                    model.eval()    # 保证模型参数不变
                    out = model(x)
                    loss = loss_fn(out, y)
                    test_losses[index] += loss.item()
        for i in range(num_opt):
            train_losses[i] /= len(train_loader)    # 求取每个batch的损失的均值
            test_losses[i] /= len(test_loader)
            train_losses_his[i].append(train_losses[i])
            test_losses_his[i].append(test_losses[i])
    return train_losses_his, test_losses_his

def plot(train_losses_his, test_losses_his, num_epochs, opt_labels):
    for i, his in enumerate(train_losses_his):
        plt.plot(range(num_epochs), his, label=opt_labels[i])
    plt.legend(loc="best")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Training Loss vs. Epoch")
    plt.show()
    for i, his in enumerate(test_losses_his):
        plt.plot(range(num_epochs), his, label=opt_labels[i])
    plt.legend(loc="best")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Test Loss vs. Epoch")
    plt.show()
if __name__ == '__main__':
    num_samples = 1000

    X = torch.randn(num_samples)
    Y = torch.randn(num_samples)
    Z = f(X, Y) + torch.normal(mean=0, std=1, size=(num_samples, 1))[0]  # 高斯扰动项
    # print(X.shape, Y.shape, Z.shape)
    '''torch.Size([1000])
    torch.Size([1000])
    torch.Size([1000])'''
    train_loader, test_loader = process_date(X, Y, Z)
    # for x, y in train_loader:
    #     print(x.size(), y.size())  # torch.Size([32, 2]) torch.Size([32, 1])


    # 损失函数
    loss_fn = nn.MSELoss()
    learning_rate = 0.01
    num_epochs = 50

    # 初始化模型序列
    num_opt = 6
    opt_labels = ["SGD_lr_1e-2", "SGD_With_Momentum_lr_1e-2", "AdaGrad_lr_0.1",
                  "RMSprop_lr_1e-2", "AdaDalte_lr_1", "Adam_lr_1e-2"]
    models = [Model() for _ in range(num_opt)]


    SGD = optim.SGD(models[0].parameters(), lr=learning_rate)
    SGD_With_Momentum = optim.SGD(models[1].parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    Adagrad = optim.Adagrad(models[2].parameters(), lr=0.1)
    RMSprop = optim.RMSprop(models[3].parameters(), lr=learning_rate)
    AdaDalte = optim.Adadelta(models[4].parameters(), lr=1.0)
    Adam = optim.Adam(models[5].parameters(), lr=learning_rate)
    opts = [SGD, SGD_With_Momentum, Adagrad, RMSprop, AdaDalte, Adam]

    train_losses_his, test_losses_his = train(num_epochs, train_loader, test_loader, models, opts, loss_fn, num_opt)
    plot(train_losses_his, test_losses_his, num_epochs, opt_labels)



