import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# 1, 定义算法CNN模型
class CNN(nn.Module):
    def __init__(self):
        """
        初始化CNN网络结构
        定义网络的各层组件
        """
        super(CNN, self).__init__()  # 调用父类构造函数

        # 第一个卷积层：输入通道是1（灰度图），输出通道32，卷积核大小3x3，填充1
        # padding=1 保证卷积后特征图尺寸不变 (28x28 -> 28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # 第二个卷积层：输入通道是32，输出通道64，卷积核大小3×3，填充1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # 池化层：使用2x2窗口，步长为2的最大池化
        # 作用：降低特征图尺寸，增加感受野，减少参数数量，提高模型泛化能力
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # dropout 层防止过拟合: 随机丢弃一部分神经元，减少神经元间的协同适应
        self.dropout1 = nn.Dropout(0.25)  # 25%的神经元会被随机丢弃
        self.dropout2 = nn.Dropout(0.5)  # 50%的神经元会被随机丢弃

        # 全连接层: 将卷积层提取的特征映射到最终输出
        # 输入维度: 64 * 7 * 7 (经过两次池化后，28x28 -> 14x14 -> 7x7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入64*7*7个特征，输出128个特征
        self.fc2 = nn.Linear(128, 10)  # 输入128个特征，输出10个类别（0-9数字）

    def forward(self, x):
        """
        定义前向传播过程
        :param x: 输入张量，形状为(batch_size, 1, 28, 28)
        :return: 输出张量，形状为(batch_size, 10)
        """
        # 第一个卷积块：卷积->ReLU激活->池化
        # x形状: (batch_size, 1, 28, 28) -> (batch_size, 32, 28, 28) -> (batch_size, 32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))

        # 第二个卷积块：卷积->ReLU激活->池化
        # x形状: (batch_size, 32, 14, 14) -> (batch_size, 64, 14, 14) -> (batch_size, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))

        # 应用dropout
        x = self.dropout1(x)

        # Flatten 操作: 将4D张量转换为2D张量，为全连接层做准备
        # x形状: (batch_size, 64, 7, 7) -> (batch_size, 64*7*7=3136)
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层 + ReLU激活
        # x形状: (batch_size, 3136) -> (batch_size, 128)
        x = F.relu(self.fc1(x))

        # 应用dropout
        x = self.dropout2(x)

        # 输出层: 不使用激活函数，因为CrossEntropyLoss内部会处理
        # x形状: (batch_size, 128) -> (batch_size, 10)
        x = self.fc2(x)

        return x


# 2, 数据预处理
# Compose函数将多个数据预处理操作组合在一起
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或numpy数组转换为PyTorch张量，并自动缩放到[0,1]范围
    transforms.Normalize((0.1307,), (0.3081,)),  # 数据标准化: 减去均值0.1307，除以标准差0.3081
])

# 3，下载并加载数据集
# MNIST手写数字数据集，包含60000个训练样本和10000个测试样本
train_dataset = datasets.MNIST(
    root='./data',  # 数据存储路径
    train=True,  # 加载训练集
    download=True,  # 如果数据不存在，自动下载
    transform=transform  # 应用定义的数据预处理
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,  # 加载测试集
    download=True,
    transform=transform
)

# 创建数据加载器
# DataLoader负责从数据集中按批次加载数据，支持自动批处理、打乱数据和多进程加载
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,  # 每个批次的样本数量
    shuffle=True  # 每个epoch开始时打乱数据顺序
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,  # 测试时使用更大的批次以提高效率
    shuffle=False  # 测试时不需要打乱数据
)

# 4, 检测是否有可用GPU
# 如果有GPU可用，使用GPU加速计算；否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 5，初始化模型、损失函数和优化器
model = CNN().to(device)  # 创建CNN模型实例，并移动到相应设备(GPU/CPU)

# 交叉熵损失函数：适用于多分类问题，内部结合了LogSoftmax和NLLLoss
criterion = nn.CrossEntropyLoss()

# Adam优化器：自适应学习率优化算法，结合了AdaGrad和RMSProp的优点
optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr: 学习率


# 6, 训练函数
def train(model, device, train_loader, optimizer, epoch):
    """
    训练模型一个epoch
    :param model: 要训练的模型
    :param device: 训练设备 (GPU/CPU)
    :param train_loader: 训练数据加载器
    :param optimizer: 优化器
    :param epoch: 当前epoch编号
    :return: 平均训练损失和准确率
    """
    model.train()  # 设置模型为训练模式（启用dropout等）
    total_loss = 0  # 累计损失
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数

    # 遍历训练数据的所有批次
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到相应设备(GPU/CPU)
        data, target = data.to(device), target.to(device)

        # 清零梯度：PyTorch会累积梯度，所以在每次反向传播前需要清零
        optimizer.zero_grad()

        # 正向传播：通过模型计算预测输出
        output = model(data)

        # 计算损失：比较预测输出和真实标签
        loss = criterion(output, target)

        # 反向传播：计算梯度
        loss.backward()

        # 更新参数：根据梯度调整模型参数
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

        # 计算准确率
        _, predicted = output.max(1)  # 获取预测类别（最大值的索引）
        total += target.size(0)  # 累计样本总数
        correct += predicted.eq(target).sum().item()  # 累计正确预测数

        # 每100个批次打印一次训练进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # 计算整个epoch的平均损失和准确率
    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


# 7, 测试函数
def test(model, device, test_loader):
    """
    测试模型性能
    :param model: 要测试的模型
    :param device: 测试设备 (GPU/CPU)
    :param test_loader: 测试数据加载器
    :return: 平均测试损失和准确率
    """
    model.eval()  # 设置模型为评估模式（禁用dropout等）
    test_loss = 0
    correct = 0
    total = 0

    # 在测试时不计算梯度，节省内存和计算资源
    with torch.no_grad():
        # 遍历测试数据的所有批次
        for data, target in test_loader:
            # 将数据移动到相应设备(GPU/CPU)
            data, target = data.to(device), target.to(device)

            # 正向传播：通过模型计算预测输出
            output = model(data)

            # 计算损失
            test_loss += criterion(output, target).item()

            # 计算准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    # 计算整个测试集的平均损失和准确率
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    # 打印测试结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


# 8, 训练和测试循环
epochs = 5  # 训练轮数
train_losses = []  # 存储每个epoch的训练损失
train_accs = []  # 存储每个epoch的训练准确率
test_losses = []  # 存储每个epoch的测试损失
test_accs = []  # 存储每个epoch的测试准确率

# 循环训练和测试模型
for epoch in range(1, epochs + 1):
    # 训练一个epoch
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)

    # 测试模型性能
    test_loss, test_acc = test(model, device, test_loader)

    # 记录当前epoch的结果
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

# 9, 绘制训练和测试曲线
plt.figure(figsize=(12, 5))  # 创建画布，设置大小

# 绘制损失曲线
plt.subplot(1, 2, 1)  # 创建1行2列的子图，选择第1个
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')  # x轴标签
plt.ylabel('Loss')  # y轴标签
plt.legend()  # 显示图例

# 绘制准确率曲线
plt.subplot(1, 2, 2)  # 选择第2个子图
plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_accs, label='Test Accuracy')
plt.xlabel('Epochs')  # x轴标签
plt.ylabel('Accuracy (%)')  # y轴标签
plt.legend()  # 显示图例

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()  # 显示图像


# 10, 可视化一些测试结果
def visualize_predictions(model, device, test_loader, num_images=10):
    """
    可视化模型在测试集上的预测结果
    :param model: 训练好的模型
    :param device: 设备 (GPU/CPU)
    :param test_loader: 测试数据加载器
    :param num_images: 要可视化的图像数量
    """
    model.eval()  # 设置模型为评估模式
    images_so_far = 0  # 已处理图像计数
    fig = plt.figure(figsize=(15, 5))  # 创建画布

    # 不计算梯度，节省资源
    with torch.no_grad():
        # 遍历测试数据
        for data, target in test_loader:
            # 将数据移动到相应设备
            data, target = data.to(device), target.to(device)

            # 获取模型预测
            output = model(data)
            pred = output.argmax(dim=1)  # 获取预测类别

            # 遍历当前批次中的每个样本
            for i in range(len(data)):
                # 如果已达到要可视化的图像数量，停止处理
                if images_so_far >= num_images:
                    plt.tight_layout()
                    plt.show()
                    return

                images_so_far += 1
                # 创建子图
                ax = fig.add_subplot(2, num_images // 2, images_so_far)
                ax.axis('off')  # 不显示坐标轴
                # 设置子图标题：真实标签和预测标签
                ax.set_title(f'True: {target[i].item()}\nPred: {pred[i].item()}')

                # 将图像数据从GPU移回CPU，并转换为numpy数组
                img = data[i].cpu().numpy().squeeze()
                # 显示图像（灰度图）
                ax.imshow(img, cmap='gray_r')

    plt.tight_layout()
    plt.show()


# 可视化一些预测结果
visualize_predictions(model, device, test_loader)