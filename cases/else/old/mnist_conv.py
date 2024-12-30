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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2D(1, 3, (5, 5))
        self.pool1 = MaxPooling2D(3, 3, 2)
        self.conv2 = Conv2D(3, 3, (3, 3))
        self.pool2 = MaxPooling2D(3, 3, 2)

        # 明确指定全连接层的输入大小
        self.fc = nn.Linear(3 * 4 * 4, 10)  # 3 是通道数，4×4 是最终特征图的宽和高

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # 不再动态初始化，直接展平
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # 展平为 (batch_size, 3*4*4)

        x = self.fc(x)
        x = F.log_softmax(x)
        return x

model = SimpleCNN()
print("Model initialized.")

criterion = NLLLoss()
optimizer = Adam(model.parameters(), lr=0.01)

def train(epoch):
    start_time = time()
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
    print(f"Epoch {epoch + 1} training time: {time() - start_time:.2f} seconds")

def test():
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
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    print(f"Testing time: {time() - start_time:.2f} seconds")

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

# SGD log：准确率83左右
# [1,  300] loss:1.854
# [1,  600] loss:0.871
# [1,  900] loss:0.731
# Epoch 1 training time: 20.62 seconds
# Accuracy on test set:71 %
# Testing time: 3.10 seconds
# [2,  300] loss:0.674
# [2,  600] loss:0.632
# [2,  900] loss:0.596
# Epoch 2 training time: 20.26 seconds
# Accuracy on test set:81 %
# ......
# [10,  600] loss:0.497
# [10,  900] loss:0.495
# Epoch 10 training time: 22.08 seconds
# Accuracy on test set:83 %
# Testing time: 3.18 seconds

# Adam log：准确率80左右
# [1,  300] loss:1.251
# [1,  600] loss:0.812
# [1,  900] loss:0.748
# Epoch 1 training time: 20.90 seconds
# Accuracy on test set:78 %
# Testing time: 3.13 seconds
# [2,  300] loss:0.711
# [2,  600] loss:0.712
# [2,  900] loss:0.712
# Epoch 2 training time: 22.62 seconds
# Accuracy on test set:79 %
# Testing time: 3.56 seconds
# ......
# [8,  300] loss:0.685
# [8,  600] loss:0.688
# [8,  900] loss:0.676
# Epoch 8 training time: 20.58 seconds
# [8,  600] loss:0.688
# [8,  900] loss:0.676
# Epoch 8 training time: 20.58 seconds
# Accuracy on test set:79 %

# 与Pytorch比较：那边准确度可以到95，Epoch training time只需一半时间（10s）
