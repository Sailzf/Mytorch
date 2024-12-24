import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from time import time

# 超参数
batch_size = 64
learning_rate = 0.01
num_epochs = 10

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)  # 输入通道 1，输出通道 3，卷积核 5×5
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3)  # 输入通道 3，输出通道 3，卷积核 3×3
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc = nn.Linear(3 * 4 * 4, 10)  # 展平后输入全连接层，输出为 10 类

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 3 * 4 * 4)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # 计算对数概率

# 初始化模型
model = SimpleCNN()
print("Model initialized.")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练函数
def train(epoch):
    model.train()  # 设为训练模式
    start_time = time()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f"[Epoch {epoch + 1}, Batch {batch_idx + 1}] Loss: {running_loss / 300:.3f}")
            running_loss = 0.0
    end_time = time()
    print(f"Epoch {epoch + 1} training time: {end_time - start_time:.2f} seconds")

# 测试函数
def test():
    model.eval()  # 设为评估模式
    start_time = time()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)  # 预测值为最大概率的类别
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    end_time = time()
    print(f"Testing time: {end_time - start_time:.2f} seconds")

# 主程序
if __name__ == '__main__':
    for epoch in range(num_epochs):
        train(epoch)
        test()
