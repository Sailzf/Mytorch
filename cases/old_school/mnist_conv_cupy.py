from torchvision import transforms, datasets
import cupy as cp
from time import time
import nvtx  # 添加这一行

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

import os
print("Process ID:", os.getpid())
# 延迟，方便附加调试
# time.sleep(10)

def prepare_mnist_data(mnist_dataset):
    data, targets = [], []
    for x, y in mnist_dataset:
        data.append(cp.array(x.numpy()))
        targets.append(y)

    data = cp.stack(data)
    targets = cp.array(targets)

    data = Tensor(data, requires_grad=False)
    targets = Tensor(targets, requires_grad=False)

    return MNISTDataset(data, targets)

# 加载和准备数据
with nvtx.annotate("Initialize Dataset", color="yellow"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST(root='data/mnist/', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)
    print("prepare mnist_train")
    train_dataset = prepare_mnist_data(mnist_train)
    print("prepare mnist_test")
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
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
model.to(device)
print("Model initialized.")

criterion = NLLLoss()
optimizer = Adam(model.parameters(), lr=0.01)

def train(epoch):
    with nvtx.annotate(f"Train Epoch {epoch}", color="green"):
        start_time = time()
        running_loss = 0.0
        for batch_idx, (inputs, target) in enumerate(train_loader):
            with nvtx.annotate("Train Batch", color="lime"):
                with nvtx.annotate("Forward Pass", color="cyan"):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                
                with nvtx.annotate("Backward Pass", color="yellow"):
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
    with nvtx.annotate("Test", color="red"):
        start_time = time()
        correct = 0
        total = 0
        with no_grad():
            for inputs, labels in test_loader:
                with nvtx.annotate("Test Batch", color="pink"):
                    with nvtx.annotate("Forward Pass", color="orange"):
                        outputs = model(inputs)
                        predicted = mymax().forward(outputs.data, axis=1)
                        predicted = cp.round(predicted)
                    
                    total += labels.array().size
                    correct += (predicted == labels.array()).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))
        print(f"Testing time: {time() - start_time:.2f} seconds")

if __name__ == '__main__':
    with nvtx.annotate("Training Loop", color="blue"):
        for epoch in range(3):
            train(epoch)
            test()

# (torch) PS D:\Users\19131\Desktop\base\Mytorch_distributed+data> python -u "d:\Users\19131\Desktop\base\Mytorch_distributed+data\mnist_conv_cupy.py"
# Model initialized.
# [1,  300] loss:1.389
# [1,  600] loss:0.622
# [1,  900] loss:0.566
# Epoch 1 training time: 25.61 seconds
# Accuracy on test set: 84 %
# Testing time: 1.47 seconds
# [2,  300] loss:0.515
# [2,  600] loss:0.536
# [2,  900] loss:0.519
# Epoch 2 training time: 24.51 seconds
# Accuracy on test set: 85 %
# Testing time: 2.30 seconds
# [3,  300] loss:0.522
# [3,  600] loss:0.525
# [3,  900] loss:0.515
# Epoch 3 training time: 26.04 seconds
# Accuracy on test set: 84 %
# Testing time: 2.87 seconds
# [4,  300] loss:0.513
# [4,  600] loss:0.513
# [4,  900] loss:0.509
# Epoch 4 training time: 25.01 seconds
# Accuracy on test set: 85 %
# Testing time: 2.48 seconds
# [5,  300] loss:0.516
# [5,  600] loss:0.508
# [5,  900] loss:0.525
# Epoch 5 training time: 18.13 seconds
# Accuracy on test set: 84 %
# Testing time: 1.17 seconds
