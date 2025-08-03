# _*_ coding: utf-8 _*_
'''
时间:      2025/8/3 23:16
@author:  andinm
'''
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

def creat_data(num_samples):
    torch.manual_seed(42)  # 设置随机种子
    x_train = torch.unsqueeze(torch.linspace(-1, 1, num_samples), dim=1)
    y_train = x_train + 0.3 * torch.randn(num_samples, 1)
    x_test = torch.unsqueeze(torch.linspace(-1, 1, num_samples), dim=1)
    y_test = x_test + 0.3 * torch.randn(num_samples, 1)
    # plt.scatter(x_train.numpy(), y_train.numpy())
    # plt.scatter(x_test.numpy(), y_test.numpy())
    # plt.show()
    return x_train, y_train, x_test, y_test
# 定义模型  简单搭建模型使用Sequential，不去定义class
def creat_model(hidden_size):
    # 可能会过拟合的网络
    net_overfitting = torch.nn.Sequential(
        torch.nn.Linear(1, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 1),
    )
    # 包含dropout网络
    net_dropout = torch.nn.Sequential(
        torch.nn.Linear(1, hidden_size),
        torch.nn.Dropout(0.5), # p = 0.5
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Dropout(0.5),  # p = 0.5
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 1),
    )
    return net_overfitting, net_dropout

# 优化器和损失函数
def loss_function():
    return torch.nn.MSELoss()  # 常用于回归问题


def get_optimizer(model, learning_rate=0.001):
    return optim.Adam(model.parameters(), lr=learning_rate)

def train(net_overfitting, net_dropout, x_train, y_train, num_epochs):
    optimizer_overfitting = get_optimizer(net_overfitting)
    optimizer_dropout = get_optimizer(net_dropout)
    loss_fc = loss_function()
    for epoch in range(num_epochs):
        pred_overfitting = net_overfitting.forward(x_train)
        loss_overfitting = loss_fc(pred_overfitting, y_train)
        optimizer_overfitting.zero_grad()
        loss_overfitting.backward()
        optimizer_overfitting.step()

        # dropout
        pred_dropout = net_dropout.forward(x_train)
        loss_dropout = loss_fc(pred_dropout, y_train)
        optimizer_dropout.zero_grad()
        loss_dropout.backward()
        optimizer_dropout.step()
def test(net_overfitting, net_dropout, x_train, y_train, x_test, y_test):
    net_overfitting.eval()
    net_dropout.eval()
    with torch.no_grad():
        test_pred_overfitting = net_overfitting.forward(x_test)
        test_pred_dropout = net_dropout.forward(x_test)
        plt.scatter(x_train, y_train, c='r', alpha=0.3, label='training')
        plt.scatter(x_test, y_test, c='b', alpha=0.3, label='testing')
        plt.plot(x_test, test_pred_overfitting.data.numpy(), "r-", lw=2, label='overfitting')
        plt.plot(x_test, test_pred_dropout.data.numpy(), "b--", lw=2, label='dropout')
        plt.legend()
        plt.ylim((-2, 2))
        plt.show()


def main():
    NUM_EPOCHS = 500
    HIDDEN_SIZE = 200
    LEARNING_RATE = 0.01
    NUM_SAMPLES = 20
    x_train, y_train, x_test, y_test = creat_data(NUM_SAMPLES)
    net_overfitting, net_dropout = creat_model(HIDDEN_SIZE)
    train(net_overfitting, net_dropout, x_train, y_train, NUM_EPOCHS)
    test(net_overfitting, net_dropout, x_train, y_train, x_test, y_test)

if __name__ == '__main__':

    main()


