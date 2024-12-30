from torchvision import transforms, datasets
import cupy as cp
from time import time
import os
import nvtx
import swanlab
from tqdm import tqdm

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

def prepare_mnist_data(mnist_dataset, cache_file, batch_size=1000):
    # 如果缓存文件存在，直接加载
    if os.path.exists(cache_file + '_data.npy') and os.path.exists(cache_file + '_targets.npy'):
        print(f"Loading cached dataset from {cache_file}")
        with tqdm(total=2, desc="Loading cache") as pbar:
            data = cp.load(cache_file + '_data.npy')
            pbar.update(1)
            targets = cp.load(cache_file + '_targets.npy')
            pbar.update(1)
        print("Cache loaded successfully")
    else:
        print("Converting dataset and creating cache...")
        # 计算总批次数
        total_samples = len(mnist_dataset)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        # 预分配数组
        sample_shape = cp.array(mnist_dataset[0][0].numpy()).shape
        data = cp.empty((total_samples, *sample_shape), dtype=cp.float32)
        targets = cp.empty(total_samples, dtype=cp.int64)
        
        # 批量处理数据
        with tqdm(total=num_batches, desc="Processing batches") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                
                # 处理当前批次
                batch_data = []
                batch_targets = []
                for idx in range(start_idx, end_idx):
                    x, y = mnist_dataset[idx]
                    batch_data.append(x.numpy())
                    batch_targets.append(y)
                
                # 转换为CuPy数组并存储
                batch_data = cp.array(batch_data)
                batch_targets = cp.array(batch_targets)
                
                # 保存到预分配的数组
                data[start_idx:end_idx] = batch_data
                targets[start_idx:end_idx] = batch_targets
                
                # 清理临时变量
                del batch_data
                del batch_targets
                cp.get_default_memory_pool().free_all_blocks()
                
                pbar.update(1)
        
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # 保存缓存
        print("\nSaving cache to disk...")
        with tqdm(total=2, desc="Saving cache") as pbar:
            cp.save(cache_file + '_data.npy', data)
            pbar.update(1)
            cp.save(cache_file + '_targets.npy', targets)
            pbar.update(1)
        print(f"Dataset cached to {cache_file}")
        
        # 清理内存
        cp.get_default_memory_pool().free_all_blocks()

    data = Tensor(data, requires_grad=False)
    targets = Tensor(targets, requires_grad=False)

    return MNISTDataset(data, targets)

# 初始化 SwanLab
run = swanlab.init(
    project="MNIST-SqueezeNet",
    experiment_name="MNIST-SqueezeNet-CuPy",
    config={
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 20,
        "device": "cuda" if cuda.is_available() else "cpu",
        "version": "1.1",  # SqueezeNet版本
        "base_channels": 64,  # 基础通道数
    },
)

# 加载和准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 设置缓存目录
cache_dir = 'data/mnist/processed_squeezenet'
train_cache = os.path.join(cache_dir, 'train')
test_cache = os.path.join(cache_dir, 'test')

mnist_train = datasets.MNIST(root='data/mnist/', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)

train_dataset = prepare_mnist_data(mnist_train, train_cache)
test_dataset = prepare_mnist_data(mnist_test, test_cache)

train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=run.config.batch_size, shuffle=False)

class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        # Squeeze层：1x1卷积减少通道数
        self.squeeze = Conv2D(in_channels, squeeze_channels, kernel_size=(1, 1))
        
        # Expand层：并行的1x1和3x3卷积
        self.expand1x1 = Conv2D(squeeze_channels, expand1x1_channels, kernel_size=(1, 1))
        self.expand3x3 = Conv2D(squeeze_channels, expand3x3_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # Squeeze
        x = F.relu(self.squeeze(x))
        
        # Expand
        x1 = F.relu(self.expand1x1(x))
        x3 = F.relu(self.expand3x3(x))
        
        # 在通道维度上拼接
        return F.concat([x1, x3], axis=1)

class SqueezeNet(nn.Module):
    def __init__(self, version='1.1', num_classes=10, base_channels=64):
        super(SqueezeNet, self).__init__()
        
        if version == '1.0':
            self.features = nn.Sequential(
                Conv2D(1, base_channels, kernel_size=(7, 7), stride=2),  # 调整输入通道为1
                MaxPooling2D(3, 3, 2),
                Fire(base_channels, 16, 64, 64),
                Fire(128, 16, 64, 64),
                MaxPooling2D(3, 3, 2),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                MaxPooling2D(3, 3, 2),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:  # version 1.1
            self.features = nn.Sequential(
                Conv2D(1, base_channels//2, kernel_size=(3, 3), stride=2),  # 调整输入通道为1
                MaxPooling2D(3, 3, 2),
                Fire(base_channels//2, 16, 64, 64),
                Fire(128, 16, 64, 64),
                MaxPooling2D(3, 3, 2),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                MaxPooling2D(3, 3, 2),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        
        # Final layers
        self.classifier = nn.Sequential(
            Conv2D(512, num_classes, kernel_size=(1, 1)),
            MaxPooling2D(13, 13, 1)  # Global average pooling
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return F.log_softmax(x)

model = SqueezeNet(version=run.config.version, base_channels=run.config.base_channels)
device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
model.to(device)

# 打印模型信息
total_params = sum(p.array().size for p in model.parameters())
print("\nModel Summary:")
print(f"Total Parameters: {total_params:,}")
print(f"Training Device: {device}")
print(f"Starting training for {run.config.num_epochs} epochs...")

criterion = NLLLoss()
optimizer = Adam(model.parameters(), lr=run.config.learning_rate)

def train(epoch):
    with nvtx.annotate(f"Train Epoch {epoch}", color="green"):
        model.train()
        start_time = time()
        running_loss = 0.0
        total_batches = len(train_loader)
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch + 1}/{run.config.num_epochs}")
        print(f"Training on {len(train_dataset)} samples with batch size {run.config.batch_size}")
        
        for batch_idx, (inputs, target) in enumerate(train_loader):
            with nvtx.annotate("Train Batch", color="lime"):
                progress = (batch_idx + 1) / total_batches * 100
                
                with nvtx.annotate("Forward Pass", color="cyan"):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)
                
                with nvtx.annotate("Backward Pass", color="yellow"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                
                # 计算准确率
                predicted = mymax().forward(outputs.data, axis=1)
                total += target.array().size
                correct += (predicted == target.array()).sum().item()
                
                if batch_idx % 10 == 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    accuracy = 100 * correct / total
                    
                    swanlab.log({
                        "train/loss": avg_loss,
                        "train/accuracy": accuracy,
                    }, step=epoch * total_batches + batch_idx)
                    
                    print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}% "
                          f"Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%", end="")
        
        # 打印epoch总结
        epoch_loss = running_loss / total_batches
        epoch_accuracy = 100 * correct / total
        epoch_time = time() - start_time
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Accuracy: {epoch_accuracy:.2f}%")
        print(f"Time Used: {epoch_time:.2f} seconds")
        print(f"Samples/second: {len(train_dataset) / epoch_time:.2f}")

        swanlab.log({
            "train/epoch_loss": epoch_loss,
            "train/epoch_accuracy": epoch_accuracy,
            "train/epoch_time": epoch_time,
            "train/samples_per_second": len(train_dataset) / epoch_time
        }, step=epoch)

def test(epoch):
    with nvtx.annotate("Test", color="red"):
        model.eval()
        start_time = time()
        total_loss = 0
        correct = 0
        total = 0
        total_batches = len(test_loader)
        
        print("\nEvaluating on test set...")
        
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                with nvtx.annotate("Test Batch", color="pink"):
                    progress = (batch_idx + 1) / total_batches * 100
                    
                    with nvtx.annotate("Forward Pass", color="orange"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        predicted = mymax().forward(outputs.data, axis=1)
                    
                    total_loss += loss.item()
                    total += labels.array().size
                    correct += (predicted == labels.array()).sum().item()
                    
                    print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}%", end="")
        
        # 计算最终指标
        accuracy = 100 * correct / total
        avg_loss = total_loss / total_batches
        test_time = time() - start_time
        
        # 记录到SwanLab
        swanlab.log({
            "test/accuracy": accuracy,
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