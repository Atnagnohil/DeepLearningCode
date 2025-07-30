# _*_ coding: utf-8 _*_
'''
时间:      2025/7/30 15:27
@author:  andinm
'''

# 导入包
import torch
from torchvision import datasets

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tool.Read_Minist_Tool import *
import matplotlib.pyplot as plt

# 1数据
def data_load(path, batch_size=100):
    # 加载数据
    # 加载训练集和测试集
    train_data = datasets.MNIST(
        root=path,  # 存储数据的根目录
        train=True,  # 加载训练集
        download=True,  # 如果本地没有就下载
        transform=transforms.ToTensor()  # 将图像转换为 Tensor 格式
    )

    test_data = datasets.MNIST(
        root=path,  # 相同的目录
        train=False,  # 加载测试集
        download=True,
        transform=transforms.ToTensor()
    )
    # 选取部分数据
    batch_size = batch_size # 表示每批次训练模型给出batch_size大小的训练集
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True  # 打乱数据
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False  # 不打乱数据
                                              )
    return train_loader, test_loader


# 展示数据
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



# 2网络结构
# 定义MLP网络 继承nn.Module
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        '''
        :param input_size: 输入数据的维度
        :param hidden_size: 隐藏层的大小
        :param num_classes: 输出数据的类别
        '''
        # 调用父类的初始化方法
        super(MLP, self).__init__()     # 调用父类 nn.Module 的初始化函数，确保网络结构和功能完整
        # 定义第一个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 定义第三个全连接层
        self.fc3 = nn.Linear(hidden_size, num_classes)
        # 定义激活函数
        self.relu = nn.ReLU()
    # 定义forward函数
    def forward(self, x):
        # 第一层运算
        out = self.fc1(x)
        # 传给激活函数
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
# 3损失函数
def get_lossFunction():
    return nn.CrossEntropyLoss()
# 4优化器
def get_optimizer(model, learning_rate=0.001):
    '''在神经网络中，优化器的本质作用就是——用 梯度下降法（或变种） 来优化模型参数，使损失函数尽可能小。'''
    return optim.Adam(model.parameters(), lr=learning_rate)
# 5训练
def train(num_epochs, train_loader, model, criterion, optimizer, device):
    '''
    :param num_epochs: 训练轮数
    :param device: 新增参数，指定训练设备 (CPU or GPU)
    :return:
    '''
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 将数据转换成合适的维度
            images = images.reshape(-1, images.shape[-1]*images.shape[-1]).to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播优化参数
            optimizer.zero_grad()  # 将梯度清0
            loss.backward()         # 反向传播让参数收敛
            # 更新参数
            optimizer.step()
            if (i+1) % 100 == 0:  # 说明此时经过来一轮训练
                print(f"Epoch: [{epoch+1}/{num_epochs}]\tStep: [{i+1}/{len(train_loader)}]\tLoss: {loss.item():.6f}")
# 6测试
def test(test_loader, model, device):
    """
    测试模型
    :param device: 新增参数，指定测试设备 (CPU or GPU)
    """
    with torch.no_grad():
        correct = 0
        total = 0
        # 从test_loader循环读取数据
        for images, labels in test_loader:
            images = images.reshape(-1, images.shape[-1] * images.shape[-1]).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # 取出最大值的索引 # 获取预测结果中概率最高的类别
            _, predicted = torch.max(outputs.data, 1)
            # 累加label
            total += labels.size(0)
            # 预测值和label值比对  获取预测正确的数目
            correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}")
# 7保存
def save(model, path):
    torch.save(model, path)

def save2(model, path):
    """保存模型的状态字典"""
    # 推荐保存模型的状态字典，而不是整个模型
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")

    # 展示部分数据
    train_loader, test_loader = data_load(r"D:\python\PycharmProjects\DeepLearningProject\data\mnist_1")
    # print(len(train_loader))  # 600 一共分成了600批次 每批次100
    # for images, labels in train_loader:
    #     print(images.shape)
    #     print(labels.shape)
    #
    #     show_data(6, images, labels)
    #     '''torch.Size([100, 1, 28, 28])
    #         torch.Size([100])
    #         '''
    #     break

    # 创建神经网络
    input_size = 28*28
    hidden_size = 512
    num_classes = 10
    model = MLP(input_size, hidden_size, num_classes).to(device)

    # 创建损失函数
    criterion = get_lossFunction()
    # 创建优化器
    optimizer = get_optimizer(model=model, learning_rate=0.001)

    #训练模型
    print("开始训练...")
    num_epochs = 10 # 设置训练轮数        对于当前数据一共10轮训练
    train(num_epochs, train_loader, model, criterion, optimizer, device)
    print("训练完成！")

    #预测
    print("\n开始测试")
    print("<UNK>...")
    test(test_loader, model, device)

    #保存
    save(model, r"D:\python\PycharmProjects\DeepLearningProject\models\1.1_model.pkl")



