# _*_ coding: utf-8 _*_
'''
时间:      2025/8/5 13:33
@author:  andinm
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
# 数据
LEARNING_RATE = 1e-3
num_epochs = 100
np.random.seed(42)
x = np.linspace(-5, 5, 100)
y =2*x + 3 + np.random.normal(0, 0.1, 100)
# 转换成张量
x = torch.unsqueeze(torch.from_numpy(x), dim=1).float()
y = torch.unsqueeze(torch.from_numpy(y), dim=1).float()
# 初始化参数
w = torch.zeros(1, requires_grad=True)
b= torch.zeros(1, requires_grad=True)
# 创建优化器
optimizer = torch.optim.Adagrad([w, b], lr=LEARNING_RATE)
losses = []
# 训练
for epoch in range(100):
    y_pred = w*x + b
    # 计算损失
    loss = torch.mean((y_pred - y)**2)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# 展示
plt.plot(range(num_epochs), losses)
plt.title("AdaGrad")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
