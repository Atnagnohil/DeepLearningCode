# _*_ coding: utf-8 _*_
'''
时间:      2025/8/2 20:05
@author:  andinm
'''
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import os
# 加载数据
def load_data(path):
    # 如果路径不存在，则创建它
    if not os.path.exists(path):
        print(f"数据路径不存在，正在创建: {path}")
        os.makedirs(path)

    train_data = torchvision.datasets.MNIST(
        root = path,
        train = True,
        download = True,
        transform = torchvision.transforms.ToTensor()
    )
    test_data = torchvision.datasets.MNIST(
        root=path,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    batch_size = 100  # 每批量选取100个数据
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True  # 打乱数据
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader

def show_data(num, data, labels):
    """
    num: 要显示的图像数量
    data: 一批图像 Tensor，shape: [batch_size, 1, 28, 28]
    labels: 一批标签 Tensor，shape: [batch_size]
    """
    plt.figure(figsize=(10, 4))
    for i in range(num):
        plt.subplot(2, (num + 1) // 2, i + 1)
        plt.imshow(data[i][0].numpy(), cmap='gray')  # data[i] 是 [1,28,28]，取第0个通道变为[28,28]
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 构建网络
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        logits = self.linear(x)
        return logits


# 损失函数
def loss_function():
    return nn.CrossEntropyLoss()
# 优化器
def get_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train(model, train_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  # i 从 0 到 600
            # 将数据放到GPU上面
            images, labels = images.reshape(-1, images.shape[-1]**2).to(device), labels.to(device)
            # 前向传播
            output = model(images)
            # 计算损失
            loss = criterion(output, labels)
            # 反向传播
            optimizer.zero_grad()   # 梯度请0
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:  # 说明此时经过来一轮训练
                print(f"Epoch: [{epoch+1}/{num_epochs}]\tStep: [{i+1}/{len(train_loader)}]\tLoss: {loss.item():.6f}")

# 评估
def test(model, test_loader, device):
    model.eval()    # 不再更新参数
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.reshape(-1, images.shape[-1]**2).to(device), labels.to(device)
            output = model(images)

            _, predicted = torch.max(output.data, 1) # 取出概率最大的，置为1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}")


def main(device):
    # 【修改】动态计算项目路径，以适应在src目录中运行脚本的情况
    # __file__ 是当前脚本的路径
    # os.path.dirname(__file__) 是脚本所在的目录 (即 src/)
    # os.path.dirname(...) 再上一层，就是项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 使用 os.path.join 来构建跨平台兼容的路径
    data_path = os.path.join(project_root, r'data\mnist_1')
    print(f"项目根目录: {project_root}")
    print(f"数据集路径: {data_path}")

    train_loader, test_loader = load_data(data_path)

    # for images, labels in train_loader:
    #     print(images.shape)
    #     print(labels.shape)
    #
    #     show_data(6, images, labels)
    #     '''torch.Size([100, 1, 28, 28])
    #         torch.Size([100])
    #         '''
    #     break
    INPUT_SIZE = 28 * 28
    OUTPUT_SIZE = 10  # 0 - 9 10个数字
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.01
    model = Model(INPUT_SIZE, OUTPUT_SIZE).to(device)
    criterion = loss_function()
    optimizer = get_optimizer(model, LEARNING_RATE)
    train(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    test(model, test_loader, device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"<当前使用设备>{device}")
    main(device)


