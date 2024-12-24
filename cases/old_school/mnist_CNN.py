from torchvision import transforms, datasets
import numpy as np
from time import time


from mytorch.ops import Max as mymax

from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader
import mytorch.module as nn  # 网络模块
from mytorch.module import Module, Linear, Conv2D, MaxPooling2D
import mytorch.functions as F  # 激活函数及其他运算
from mytorch.functions import relu, softmax
from mytorch.optim import Adam, SGD, Adagrad  # 优化器
from mytorch.loss import CrossEntropyLoss, NLLLoss  # 损失函数
from mytorch import cuda

def prepare_mnist_data(mnist_dataset):
    data, targets = [], []
    for x, y in mnist_dataset:
        data.append(np.array(x))
        targets.append(y)

    data = np.stack(data)
    targets = np.array(targets)

    data = Tensor(data, requires_grad=False)
    targets = Tensor(targets, requires_grad=False)

    return MNISTDataset(data, targets)

# 加载和准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(root='data/mnist/', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)

train_dataset = prepare_mnist_data(mnist_train)
test_dataset = prepare_mnist_data(mnist_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = Conv2D(1, 32, (3, 3), padding=1)  # 增加filter数量，减小kernel size
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = MaxPooling2D(2, 2, 2)  # 使用更小的池化窗口
        self.dropout1 = nn.Dropout(0.25)

        # 第二个卷积块
        self.conv2 = Conv2D(32, 64, (3, 3), padding=1)  # 增加filter数量
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = MaxPooling2D(2, 2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # 第三个卷积块
        self.conv3 = Conv2D(64, 64, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = MaxPooling2D(2, 2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # 全连接层
        self.fc1 = nn.Linear(64 * 3 * 3, 512)  # 增加神经元数量
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # 全连接层
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return F.log_softmax(x)

model = ImprovedCNN()
print("Model initialized.")

criterion = NLLLoss()
optimizer = Adam(model.parameters(), lr=0.001)  # 降低学习率

# 添加学习率调度器
scheduler = nn.StepLR(optimizer, step_size=2, gamma=0.9)  # 每2个epoch降低10%的学习率

def train(epoch):
    model.train()  # 确保模型处于训练模式
    start_time = time()
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        nn.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

    scheduler.step()  # 更新学习率
    print(f"Epoch {epoch + 1} training time: {time() - start_time:.2f} seconds")

def test():
    model.eval()  # 确保模型处于评估模式
    start_time = time()
    correct = 0
    total = 0
    with no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = mymax().forward(outputs.data, axis=1)
            predicted = np.round(predicted)

            total += labels.array().size
            correct += (predicted == labels.array()).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on test set: %d %%' % accuracy)
    print(f"Testing time: {time() - start_time:.2f} seconds")
    return accuracy


if __name__ == '__main__':
    best_acc = 0
    for epoch in range(15):  # 增加训练轮数
        train(epoch)
        acc = test()
        if acc > best_acc:
            best_acc = acc
            # 可以在这里保存最佳模型

    print(f"Best accuracy: {best_acc}%")

# Testing time: 1.77 seconds
# [8,  300] loss:0.768
# [8,  600] loss:0.776
# [8,  900] loss:0.760
# Epoch 8 training time: 12.37 seconds
# Accuracy on test set: 75 %
# Testing time: 1.86 seconds
# [9,  300] loss:0.763
# [9,  600] loss:0.772
# [9,  900] loss:0.770
# Epoch 9 training time: 12.48 seconds
# Accuracy on test set: 76 %
# Testing time: 1.83 seconds
# [10,  300] loss:0.768
# [10,  600] loss:0.759
# [10,  900] loss:0.763
# Epoch 10 training time: 12.98 seconds
# Accuracy on test set: 77 %
# Testing time: 1.81 seconds