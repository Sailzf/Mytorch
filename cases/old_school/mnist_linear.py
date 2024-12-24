from torchvision import transforms, datasets
import numpy as np

from mytorch.tensor import Tensor
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader
import mytorch.module as nn
import mytorch.functions as F
from mytorch.optim import SGD
from mytorch.loss import NLLLoss
from mytorch.tensor import no_grad
from mytorch.ops import Max as mymax

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载原始MNIST数据
mnist_train = datasets.MNIST(root='data/mnist/', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)

def prepare_mnist_data(mnist_dataset):
    # 收集所有数据
    data, targets = [], []
    for x, y in mnist_dataset:
        data.append(np.array(x))
        targets.append(y)
    
    # 转换为numpy数组
    data = np.stack(data)
    targets = np.array(targets)
    
    # 转换为Tensor
    data = Tensor(data, requires_grad=False)
    targets = Tensor(targets, requires_grad=False)
    
    return MNISTDataset(data, targets)

# 准备数据集
train_dataset = prepare_mnist_data(mnist_train)
test_dataset = prepare_mnist_data(mnist_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Feedforward(nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.linear = nn.Linear(784, 10)
        
    def forward(self, x):
        x = x.reshape(-1, 784)  # 展平图像
        return F.log_softmax(self.linear(x))

model = Feedforward()
criterion = NLLLoss()
optimizer = SGD(model.parameters(), lr=0.01)

def train(epoch):
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs = inputs.reshape(-1, 784)
        outputs = model(inputs)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % 
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.reshape(-1, 784)
            outputs = model(inputs)
            predicted = mymax().forward(outputs.data, axis=1)
            predicted = np.round(predicted)
            
            labels = labels.array()
            total += labels.size
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
