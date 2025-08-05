# _*_ coding: utf-8 _*_
'''
时间:      2025/8/5 21:32
@author:  andinm
'''

import torch
import matplotlib.pyplot as plt

'''target
    寻找函数最小值
'''
def f(x, y):
    return x*x + 2*y*y + 1
def train(x, y, num_epochs, learning_rate):
    xs = []  # 记录每一步梯度下降的数值
    ys = []
    zs = []
    for epoch in range(num_epochs):
        z = f(x, y)  # 计算当前y
        xs.append(x.item())
        ys.append(y.item())
        zs.append(z.item())

        # 反向传播求梯度
        z.backward()
        with torch.no_grad():
            x -= learning_rate * x.grad
            y -= learning_rate * y.grad
            x.grad.zero_()
            y.grad.zero_()
    return xs, ys, zs
def plot1_3D(xs, ys, zs):
    # 绘制原图
    ax = plt.figure().add_subplot(projection='3d')
    # 制图下降轨迹
    ax.plot(xs, ys, zs, 'r-')
    ax.scatter(xs, ys, zs, c='r', s=50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot2_3D(xs, ys, zs):
    """
    绘制3D函数曲面和梯度下降的轨迹。
    """
    # 1. 创建画布和3D坐标系
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')
    '''x_range = np.arange(-15, 15, 0.5)
    y_range = np.arange(-20, 20, 0.5)
    # 使用 meshgrid 创建二维网格
    X, Y = np.meshgrid(x_range, y_range)'''
    # 2. 创建用于绘制曲面的网格数据
    # 定义 x 和 y 的范围，需要覆盖梯度下降的轨迹
    x_origin = torch.arange(-15, 15, 0.5)
    y_origin = torch.arange(-20, 20, 0.5)
    # 使用 meshgrid 创建二维网格
    X, Y = torch.meshgrid(x_origin, y_origin, indexing='ij')
    # 计算网格上每个点的 z 值
    Z = X ** 2 + 2 * Y ** 2 + 1

    # 3. 绘制3D函数曲面
    # 使用 plot_surface 绘制，并设置颜色映射和透明度
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # 4. 绘制梯度下降的轨迹
    # 绘制红色的轨迹线
    ax.plot(xs, ys, zs, 'r-', linewidth=2, zorder=10)
    # 绘制每一步的点，使其更加突出
    ax.scatter(xs, ys, zs, c='r', s=50, zorder=10)

    # 5. 设置坐标轴标签和标题
    ax.set_title("Gradient Descent on 3D Surface")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # 调整视角
    ax.view_init(elev=30, azim=35)

    plt.show()

def plot3_2D(xs, ys):
    '''绘制映射图像'''
    # 定义 x 和 y 的范围，需要覆盖梯度下降的轨迹
    x_origin = torch.arange(-15, 15, 0.5)
    y_origin = torch.arange(-20, 20, 0.5)
    # 使用 meshgrid 创建二维网格
    X, Y = torch.meshgrid(x_origin, y_origin, indexing='ij')
    Z = f(X, Y)
    plt.contourf(X, Y, Z, cmap='viridis', alpha=0.3)
    # 绘制搜索的二维投影
    plt.plot(xs, ys, 'r-')
    plt.scatter(xs, ys, c='r', s=50)
    plt.show()


if __name__ == '__main__':
    x = torch.tensor(-12.0, requires_grad=True)
    y = torch.tensor(-15.0, requires_grad=True)
    '''当张量设置 requires_grad=True 时，PyTorch自动记录所有相关运算'''
    learning_rate = 0.1     # 学习率设置大一点方便展示下降过程

    num_epochs = 100
    xs, ys, zs = train(x, y, num_epochs, learning_rate)
    plot1_3D(xs, ys, zs)
    plot2_3D(xs, ys, zs)
    plot3_2D(xs, ys)


