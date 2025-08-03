# _*_ coding: utf-8 _*_
'''
时间:      2025/8/3 20:34
@author:  andinm
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim


# 加载数据集
def creat_data():
    # 设置随机种子
    np.random.seed(42)
    '''生成一个包含100个随机数的数组，这些随机数均匀分布在区间
        [-5, 5)（即包含 - 5，但不包含5）之间，并将这些数组织成一个100行1列的二维数组。'''
    x = np.random.uniform(-5, 5, size=(320, 1))
    y = x ** 2 + x * 1 + 1 + 5 * np.random.normal(0, 1, size=(320, 1))  # 加入正态分布噪声
    # plt.scatter(x, y)
    #
    # plt.show()
    # 将numpy转换为浮点型的pytorch张量
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    return x_tensor, y_tensor


# 划分数据集
# 2. 使用 TensorDataset 和 random_split 来划分数据集
def split_data(x_tensor, y_tensor):
    # 将输入和标签组合成一个 TensorDataset
    dataset = TensorDataset(x_tensor, y_tensor)

    # 计算训练集和测试集的大小
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # 使用 random_split 随机划分为训练集和测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 使用 DataLoader 封装成迭代器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


# 选择合适的机器学习模型 定义三个模型来完整数据拟合，线性模型，网络模型，过度拟合的模型

'''线性回归模型'''


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


'''多层感知机'''


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 8)
        self.output = nn.Linear(8, 1)  # 只有一层网络
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.hidden(x))
        return self.output(out)


'''更加复杂的多层感知机'''


class MLPoverfit(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.hidden1(x))
        out = self.relu(self.hidden2(out))
        return self.output(out)


# 优化器和损失函数
def loss_function():
    return nn.MSELoss()  # 常用于回归问题


def get_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)


# 训练
def train_model(model, train_loader, test_loader, loss_fc, optimizer, num_epochs, device):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (data, label) in enumerate(train_loader):
            # 将数据放到GPU上面
            data, label = data.to(device), label.to(device)
            # 前向传播
            output = model(data)
            # 计算损失
            loss = loss_fc(output, label)
            train_loss += loss
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 跟新梯度
        train_loss = (train_loss / len(train_loader)).cpu().detach().item() # 将GPU上面的张量转到cpu上的列表
        train_losses.append(train_loss)

        # 在测试数据上面评估
        test_loss = test_model(model, test_loader, device, loss_fc).cpu().detach().item()
        test_losses.append(test_loss)
    return train_losses, test_losses


# 测试
def test_model(model, test_loader, device, loss_fc):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            # 计算损失
            loss = loss_fc(output, label)
            test_loss += loss
        test_loss = test_loss / len(test_loader)
    return test_loss


# 展示误差
def show_errors(models, train_loader, test_loader, num_epochs, device):
    """
    修正后的绘图函数，将每个模型的训练和测试损失画在独立的子图中。

    关键改动:
    1. 使用 plt.subplots() 一次性创建所有子图。
    2. 循环遍历每个模型和对应的子图 axes。
    3. 将 plt.show() 移动到循环外部，确保所有图都绘制完成后再显示。
    4. 调整了 plt.ylim() 的设置，以确保曲线可见。
    """
    print("开始绘图...\n")

    # 1. 创建一个包含多个子图的图表。
    # nrows=len(models) 创建与模型数量相等的行数，ncols=1 创建1列
    # figsize=(10, 5 * len(models)) 调整图表大小，确保每个子图有足够的空间
    fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(10, 5 * len(models)))

    # 2. 如果只有一个模型，axes不是数组，需要特殊处理
    if len(models) == 1:
        axes = [axes]

    # 3. 循环遍历模型和对应的子图
    for i, model in enumerate(models):
        # 获取当前子图
        ax = axes[i]

        # 定义损失函数和优化器
        loss_fc = loss_function()
        optimizer = get_optimizer(model)

        # 得到每个模型的训练误差和测试误差
        train_loss, test_loss = train_model(model, train_loader, test_loader,
                                            loss_fc, optimizer, num_epochs, device)

        # 4. 在当前的子图上进行绘图
        x = range(num_epochs)
        ax.set_title(f"model: {model.__class__.__name__}")
        ax.plot(x, train_loss, label=f"Train: {model.__class__.__name__}", color="blue")
        ax.plot(x, test_loss, label=f"Test: {model.__class__.__name__}", color="red")

        # 5. 设置y轴范围，让其自动缩放以展示完整的曲线
        # ax.set_ylim(bottom=0)

        ax.legend(loc="upper right")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)  # 添加网格线，使图表更清晰

    # 6. 调整子图之间的间距，使其不重叠
    plt.tight_layout()

    # 7. 最后，在循环外部调用 plt.show()，一次性显示所有图表
    plt.show()


def main(device):
    print("正在加载数据\n")
    x, y = creat_data()
    # print(x.shape, y.shape)

    train_loader, test_loader = split_data(x, y)
    # for i, (data, label) in enumerate(train_loader):
    #     print(i, data.shape, label.shape)
    print("数据加载完毕\n")

    models = [LinearRegression().to(device), MLP().to(device), MLPoverfit().to(device)]
    NUM_EPOCHS = 30
    show_errors(models, train_loader, test_loader, NUM_EPOCHS, device)


if __name__ == '__main__':
    # 在GPU上面训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备{device}\n")

    main(device)





