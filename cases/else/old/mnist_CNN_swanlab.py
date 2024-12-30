from torchvision import transforms, datasets
import numpy as np
import swanlab

from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader
import mytorch.module as nn
import mytorch.functions as F
from mytorch.optim import Adam
from mytorch.loss import NLLLoss
from mytorch import cuda
from mytorch.ops import Max as mymax

# 初始化 SwanLab
run = swanlab.init(
    project="MNIST-Cupy",
    experiment_name="CupyCNN",
    mode="local",
    config={
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "batch_size": 64,
        "num_epochs": 10,
        "device": "cuda" if cuda.is_available() else "cpu",
    },
)

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

train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=run.config.batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2D(1, 3, (5, 5))
        self.pool1 = MaxPooling2D(3, 3, 2)
        self.conv2 = Conv2D(3, 3, (3, 3))
        self.pool2 = MaxPooling2D(3, 3, 2)
        self.fc = nn.Linear(3 * 4 * 4, 10)  # 3 是通道数，4×4 是最终特征图的宽和高

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # 展平为 (batch_size, 3*4*4)
        x = self.fc(x)
        x = F.log_softmax(x)
        return x

model = SimpleCNN()
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
model.to(device)

criterion = NLLLoss()
optimizer = Adam(model.parameters(), lr=run.config.learning_rate)

def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 20 == 0:
            swanlab.log({"train/loss": loss.item()})
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test(epoch):
    model.eval()
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
    swanlab.log({"test/accuracy": accuracy}, step=epoch)
    print('Accuracy on test set: %d %%' % accuracy)

if __name__ == '__main__':
    for epoch in range(run.config.num_epochs):
        swanlab.log({"train/epoch": epoch + 1}, step=epoch + 1)
        train(epoch)
        test(epoch)

    print("Training completed.")
