from torchvision import transforms, datasets
import numpy as np
from time import time
import os

from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import Dataset
from mytorch.dataloader import DataLoader
import mytorch.module as nn
from mytorch.optim import Adam
from mytorch.loss import CrossEntropyLoss
from mytorch import cuda
import swanlab
from mytorch.models.mobilenet import MobileNetV1

class CIFAR10Dataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)

def prepare_cifar10_data(cifar_dataset, cache_file):
    if os.path.exists(cache_file + '_data.npy') and os.path.exists(cache_file + '_targets.npy'):
        print(f"Loading cached dataset from {cache_file}")
        data = np.load(cache_file + '_data.npy')
        targets = np.load(cache_file + '_targets.npy')
    else:
        print("Converting dataset and creating cache...")
        data, targets = [], []
        for x, y in cifar_dataset:
            data.append(np.array(x))
            targets.append(y)
        
        data = np.stack(data)
        targets = np.array(targets)
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file + '_data.npy', data)
        np.save(cache_file + '_targets.npy', targets)
        print(f"Dataset cached to {cache_file}")

    data = Tensor(data.transpose(0, 3, 1, 2) / 255.0, requires_grad=False)  # 转换为NCHW格式
    targets = Tensor(targets, requires_grad=False)
    return CIFAR10Dataset(data, targets)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 设置缓存目录
cache_dir = 'data/cifar10/processed'
train_cache = os.path.join(cache_dir, 'train')
test_cache = os.path.join(cache_dir, 'test')

# 加载CIFAR-10数据集
cifar_train = datasets.CIFAR10(root='data/cifar10/', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='data/cifar10/', train=False, download=True, transform=transform)

train_dataset = prepare_cifar10_data(cifar_train, train_cache)
test_dataset = prepare_cifar10_data(cifar_test, test_cache)

# 初始化SwanLab
run = swanlab.init(
    project="CIFAR10-MobileNet",
    experiment_name="CIFAR10-MobileNet",
    config={
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 50,
        "device": "cuda" if cuda.is_available() else "cpu",
        "width_multiplier": 0.5  # 使用较小的模型以加快训练
    },
)

train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=run.config.batch_size, shuffle=False)

# 创建MobileNet模型
model = MobileNetV1(num_classes=10, width_multiplier=run.config.width_multiplier)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=run.config.learning_rate)

def train(epoch):
    start_time = time()
    running_loss = 0.0
    total_batches = len(train_loader)
    print(f"\nEpoch {epoch + 1}/{run.config.num_epochs}")
    
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        progress = (batch_idx + 1) / total_batches * 100
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            swanlab.log({
                "train/loss": avg_loss,
            }, step=epoch * total_batches + batch_idx)
            
            print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}% "
                  f"Loss: {avg_loss:.4f}", end="")
    
    epoch_loss = running_loss / total_batches
    epoch_time = time() - start_time
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"Average Loss: {epoch_loss:.4f}")
    print(f"Time Used: {epoch_time:.2f} seconds")

    swanlab.log({
        "train/epoch_loss": epoch_loss,
        "train/epoch_time": epoch_time,
    }, step=epoch)

def test(epoch):
    start_time = time()
    correct = 0
    total = 0
    total_loss = 0
    total_batches = len(test_loader)
    
    print("\nEvaluating on test set...")
    model.eval()
    
    with no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            progress = (batch_idx + 1) / total_batches * 100
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predicted = outputs.data.argmax(axis=1)
            
            total_loss += loss.item()
            total += targets.array().size
            correct += (predicted == targets.array()).sum().item()
            
            print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}%", end="")
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / total_batches
    test_time = time() - start_time
    
    swanlab.log({
        "test/accuracy": accuracy,
        "test/loss": avg_loss,
        "test/time": test_time
    }, step=epoch)
    
    print(f"\nTest Set Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time Used: {test_time:.2f} seconds")

if __name__ == '__main__':
    total_params = sum(p.array().size for p in model.parameters())
    print(f"\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Device: {run.config.device}")
    print(f"\nStarting training for {run.config.num_epochs} epochs...")
    
    for epoch in range(run.config.num_epochs):
        train(epoch)
        test(epoch)
        
        swanlab.log({
            "train/learning_rate": optimizer.param_groups[0]["lr"]
        }, step=epoch) 