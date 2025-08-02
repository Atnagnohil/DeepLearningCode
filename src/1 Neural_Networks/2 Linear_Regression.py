# _*_ coding: utf-8 _*_
'''
时间:      2025/8/1 23:12
@author:  andinm
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
def creat_data():
    # 生成随机种子, 使得每次代码运行结果相同
    np.random.seed(42)
    # 生出数据
    x = np.random.randn(100, 1)
    '''生成一个形状为(100, 1)
    的数组，每个元素都是从标准正态分布（均值为0，标准差为1）中随机采样出来的。'''
    y = 1 + 2*x + np.random.rand(100, 1)

    # 将数据转换成pytorch tensor
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    return x_tensor, y_tensor
def loss_function(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

def train_model(x_tensor, y_tensor, w, b, device, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        # 计算预测数据
        y_pred = (x_tensor * w + b).to(device)
        loss = loss_function(y_pred, y_tensor)
        # 反向传播
        loss.backward()
        if w.grad is None:
            raise ValueError("w.grad is None after backward")

        # 在torch.no_grad()借助框架计算的梯度w.grad  b.grad ，手动更新参数 随后清空
        with torch.no_grad():
            ''' 在PyTorch中，一个张量的
            .grad属性只有在它是计算图中的 叶子节点（leaftensor）
            并且参与了反向传播（backward()）后才会被填充。'''
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            # 情况梯度
            w.grad.zero_()
            b.grad.zero_()
    return w, b
def show(x_tensor, y_tensor, w, b):
    ''' 因为x_tensor、y_tensor、w和b
        都在GPU上，直接调用.numpy()
        会报错（NumPy不支持GPU张量）。
        '''
    # 绘制散点
    plt.scatter(x_tensor.cpu().numpy(), y_tensor.cpu().numpy(), marker='o')
    # 绘制曲线
    y_pred = x_tensor * w + b

    '''y_pred 的来源在 show 函数中，y_pred 是通过 y_pred = x_tensor * w + b 计算得到的。
            w 和 b 是模型的参数，通常在训练过程中会设置为 requires_grad=True，以便计算梯度。
            当 w 和 b 需要梯度时，y_pred 作为它们的计算结果，也会继承 requires_grad=True 的属性。


            为什么 .numpy() 会报错
            PyTorch 不允许直接将带有 requires_grad=True 的张量转换为 NumPy 数组，因为这可能会干扰自动梯度计算机制。
            而 Matplotlib 的 plt.plot() 需要 NumPy 数组作为输入，因此直接调用 y_pred.cpu().numpy() 会触发这个错误。'''
    plt.plot(x_tensor.cpu().numpy(), y_pred.detach().cpu().numpy())
    plt.show()
# 主函数
def main():
    x, y = creat_data()
    x_tensor, y_tensor = x.to(device), y.to(device)  # 将数据放到GPU上面训练
    # print(x_tensor)

    # 设置超参数
    learning_rate = 1e-2
    num_epochs = 1000

    # # 初始化参数
    # w = torch.randn(1, requires_grad=True).to(device)  # 生成均值为0，标准差为1的， 大小为1的张量
    # b = torch.zeros(1, requires_grad=True).to(device)
    # 初始化参数为标量张量
    w = torch.randn((), requires_grad=True, device=device)  # 标量张量
    b = torch.zeros((), requires_grad=True, device=device)  # 标量张量

    # 训练模型
    w, b = train_model(x_tensor, y_tensor, w, b, device, learning_rate, num_epochs)
    show(x_tensor, y_tensor, w, b)
    print(f"W = {w}")
    print(f"b = {b}")

if __name__ == '__main__':

    # 使用GPU训练数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")

    main()
