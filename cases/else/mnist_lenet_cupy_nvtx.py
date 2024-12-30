from torchvision import transforms, datasets
import cupy as cp
from time import time
import os
import nvtx
import swanlab

from mytorch.ops import Max as mymax
from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader
import mytorch.module as nn
from mytorch.module import Module, Linear, Conv2D, MaxPooling2D
import mytorch.functions as F
from mytorch.functions import relu, softmax
from mytorch.optim import Adam, SGD, Adagrad
from mytorch.loss import CrossEntropyLoss, NLLLoss
from mytorch import cuda

def prepare_mnist_data(mnist_dataset, cache_file):
    # 如果缓存文件存在，直接加载
    if os.path.exists(cache_file + '_data.npy') and os.path.exists(cache_file + '_targets.npy'):
        print(f"Loading cached dataset from {cache_file}")
        data = cp.load(cache_file + '_data.npy')
        targets = cp.load(cache_file + '_targets.npy')
    else:
        print("Converting dataset and creating cache...")
        data, targets = [], []
        for x, y in mnist_dataset:
            data.append(cp.array(x.numpy()))
            targets.append(y)

        data = cp.stack(data)
        targets = cp.array(targets)
        
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # 保存为缓存文件
        cp.save(cache_file + '_data.npy', data)
        cp.save(cache_file + '_targets.npy', targets)
        print(f"Dataset cached to {cache_file}")

    data = Tensor(data, requires_grad=False)
    targets = Tensor(targets, requires_grad=False)

    return MNISTDataset(data, targets)

# 初始化 SwanLab
run = swanlab.init(
    project="MNIST-LeNet-1229",
    experiment_name="MNIST-LeNet-cupy",
    config={
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "batch_size": 64,
        "num_epochs": 10,
        "device": "cuda" if cuda.is_available() else "cpu",
    },
)

# 加载和准备数据
with nvtx.annotate("Initialize Dataset", color="yellow"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 设置缓存目录
    cache_dir = 'data/mnist/processed_cupy'
    train_cache = os.path.join(cache_dir, 'train')
    test_cache = os.path.join(cache_dir, 'test')
    
    mnist_train = datasets.MNIST(root='data/mnist/', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)
    
    train_dataset = prepare_mnist_data(mnist_train, train_cache)
    test_dataset = prepare_mnist_data(mnist_test, test_cache)
    
    train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=run.config.batch_size, shuffle=False)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层: 1个输入通道, 6个输出通道, 5x5的卷积核
        self.conv1 = Conv2D(1, 6, (5, 5))
        # 第一个池化层: 2x2
        self.pool1 = MaxPooling2D(2, 2, 2)
        # 第二个卷积层: 6个输入通道, 16个输出通道, 5x5的卷积核
        self.conv2 = Conv2D(6, 16, (5, 5))
        # 第二个池化层: 2x2
        self.pool2 = MaxPooling2D(2, 2, 2)
        # 三个全连接层
        self.fc1 = Linear(16 * 4 * 4, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

model = LeNet()
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
model.to(device)

# 打印模型信息
total_params = sum(p.array().size for p in model.parameters())
print(f"\nModel Summary:")
print(f"Total Parameters: {total_params:,}")
print(f"Training Device: {device}")
print(f"\nStarting training for {run.config.num_epochs} epochs...")

criterion = NLLLoss()
optimizer = Adam(model.parameters(), lr=run.config.learning_rate)

def train(epoch):
    with nvtx.annotate(f"Train Epoch {epoch}", color="green"):
        start_time = time()
        running_loss = 0.0
        total_batches = len(train_loader)
        print(f"\nEpoch {epoch + 1}/{run.config.num_epochs}")
        print(f"Training on {len(train_dataset)} samples with batch size {run.config.batch_size}")
        
        for batch_idx, (inputs, target) in enumerate(train_loader):
            with nvtx.annotate("Train Batch", color="lime"):
                # 计算进度百分比
                progress = (batch_idx + 1) / total_batches * 100
                
                with nvtx.annotate("Forward Pass", color="cyan"):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                
                with nvtx.annotate("Backward Pass", color="yellow"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                
                # 每10个batch更新一次损失
                if batch_idx % 10 == 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    swanlab.log({
                        "main/loss": avg_loss,
                    }, step=epoch * total_batches + batch_idx)
                    
                    # 打印进度条
                    print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}% "
                          f"Loss: {avg_loss:.4f}", end="")
        
        # 打印epoch总结
        epoch_loss = running_loss / total_batches
        epoch_time = time() - start_time
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Time Used: {epoch_time:.2f} seconds")
        print(f"Samples/second: {len(train_dataset) / epoch_time:.2f}")

        swanlab.log({
            "train/epoch_loss": epoch_loss,
            "train/epoch_time": epoch_time,
            "main/samples_per_second": len(train_dataset) / epoch_time
        }, step=epoch)

def test(epoch):
    with nvtx.annotate("Test", color="red"):
        start_time = time()
        correct = 0
        total = 0
        total_loss = 0
        total_batches = len(test_loader)
        
        print("\nEvaluating on test set...")
        
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                with nvtx.annotate("Test Batch", color="pink"):
                    # 计算进度百分比
                    progress = (batch_idx + 1) / total_batches * 100
                    
                    with nvtx.annotate("Forward Pass", color="orange"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        predicted = mymax().forward(outputs.data, axis=1)
                        predicted = cp.round(predicted)
                    
                    total_loss += loss.item()
                    total += labels.array().size
                    correct += (predicted == labels.array()).sum().item()
                    
                    # 打印进度条
                    print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}%", end="")
        
        # 计算最终指标
        accuracy = 100 * correct / total
        avg_loss = total_loss / total_batches
        test_time = time() - start_time
        
        # 记录到SwanLab
        swanlab.log({
            "main/accuracy": accuracy,
            "test/loss": avg_loss,
            "test/time": test_time
        }, step=epoch)
        
        # 打印测试结果总结
        print(f"\nTest Set Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Time Used: {test_time:.2f} seconds")
        print(f"Samples/second: {len(test_dataset) / test_time:.2f}")

if __name__ == '__main__':
    with nvtx.annotate("Training Loop", color="blue"):
        for epoch in range(run.config.num_epochs):
            train(epoch)
            test(epoch)
            
            # 每个epoch后记录学习率
            swanlab.log({
                "train/learning_rate": optimizer.param_groups[0]["lr"]
            }, step=epoch)
