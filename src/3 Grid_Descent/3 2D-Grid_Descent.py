# _*_ coding: utf-8 _*_
'''
时间:      2025/8/5 21:04
@author:  andinm
'''
import torch
import matplotlib.pyplot as plt
'''target
    寻找函数最小值'''
def f(x):
    return x*x + 4*x + 1
def train(num_epochs, x, learning_rate):
    xs = []  # 记录每一步梯度下降的数值
    ys = []
    for epoch in range(num_epochs):
        y = f(x)  # 计算当前y
        xs.append(x.item())
        ys.append(y.item())

        # 反向传播求梯度
        y.backward()
        with torch.no_grad():
            x -= learning_rate * x.grad
            x.grad.zero_()
    return xs, ys
def plot(xs, ys):
    x_origin = torch.arange(-10, 10, 0.1)
    '''功能：创建一维张量（类似NumPy的arange）start end step'''
    y_origin = f(x_origin)

    # 绘制原图
    plt.plot(x_origin, y_origin)
    # 绘制搜索过程
    plt.plot(xs, ys, 'r--')
    plt.scatter(xs, ys, s=50, c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("2D_Grid_Descent")
    plt.show()

if __name__ == '__main__':
    x = torch.tensor(-10.0, requires_grad=True)
    '''当张量设置 requires_grad=True 时，PyTorch自动记录所有相关运算'''
    learning_rate = 0.7     # 学习率设置大一点方便展示下降过程

    num_epochs = 100
    xs, ys = train(num_epochs, x, learning_rate)
    plot(xs, ys)


