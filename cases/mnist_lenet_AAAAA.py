from torchvision import transforms, datasets
import numpy as np
from time import time
import os

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
import swanlab

def prepare_mnist_data(mnist_dataset, cache_file):
    # 如果缓存文件存在，直接加载
    if os.path.exists(cache_file + '_data.npy') and os.path.exists(cache_file + '_targets.npy'):
        print(f"Loading cached dataset from {cache_file}")
        data = np.load(cache_file + '_data.npy')
        targets = np.load(cache_file + '_targets.npy')
    else:
        print("Converting dataset and creating cache...")
        data, targets = [], []
        for x, y in mnist_dataset:
            data.append(np.array(x))
            targets.append(y)

        data = np.stack(data)
        targets = np.array(targets)
        
        # 确保缓存目录存在
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # 保存为缓存文件
        np.save(cache_file + '_data.npy', data)
        np.save(cache_file + '_targets.npy', targets)
        print(f"Dataset cached to {cache_file}")

    data = Tensor(data, requires_grad=False)
    targets = Tensor(targets, requires_grad=False)

    return MNISTDataset(data, targets)

# 加载和准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 设置缓存目录
cache_dir = 'data/mnist/processed'
train_cache = os.path.join(cache_dir, 'train')
test_cache = os.path.join(cache_dir, 'test')

mnist_train = datasets.MNIST(root='data/mnist/', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform)

train_dataset = prepare_mnist_data(mnist_train, train_cache)
test_dataset = prepare_mnist_data(mnist_test, test_cache)

# 在SwanLab初始化之前添加配置
config = {
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "batch_size": 64,
    "num_epochs": 10,
    "device": "cuda" if cuda.is_available() else "cpu",
    # 新增配置项
    "lr_scheduler": {
        "patience": 3,
        "min_lr": 1e-6,
        "factor": 0.1
    },
    "early_stopping": {
        "patience": 7,
        "min_delta": 1e-4
    },
    "model": {
        "name": "LeNet",
        "conv1_channels": 6,
        "conv2_channels": 16,
        "fc1_size": 120,
        "fc2_size": 84
    },
    "training": {
        "save_best": True,
        "save_frequency": 5,  # 每多少个epoch保存一次
        "log_frequency": 10   # 每多少个batch记录一次
    }
}

# 修改SwanLab初始化
run = swanlab.init(
    project="MNIST-LeNet",
    experiment_name="MNIST-LeNet",
    config=config
)

train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=run.config.batch_size, shuffle=False)

class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()
        # 使用配置中的参数
        self.conv1 = Conv2D(1, config["model"]["conv1_channels"], (5, 5))
        self.pool1 = MaxPooling2D(2, 2, 2)
        self.conv2 = Conv2D(config["model"]["conv1_channels"], 
                           config["model"]["conv2_channels"], (5, 5))
        self.pool2 = MaxPooling2D(2, 2, 2)
        self.fc1 = Linear(config["model"]["conv2_channels"] * 4 * 4, 
                         config["model"]["fc1_size"])
        self.fc2 = Linear(config["model"]["fc1_size"], 
                         config["model"]["fc2_size"])
        self.fc3 = Linear(config["model"]["fc2_size"], 10)

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

model = LeNet(run.config)
print("Model initialized.")

criterion = NLLLoss()
optimizer = Adam(model.parameters(), lr=run.config.learning_rate)

def train(epoch):
    start_time = time()
    running_loss = 0.0
    total_batches = len(train_loader)
    print(f"\nEpoch {epoch + 1}/{run.config.num_epochs}")
    print(f"Training on {len(train_dataset)} samples with batch size {run.config.batch_size}")
    
    model.train()
    for batch_idx, (inputs, target) in enumerate(train_loader):
        progress = (batch_idx + 1) / total_batches * 100
        
        outputs = model(inputs)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 使用配置的日志频率
        if batch_idx % run.config.training.log_frequency == 0:
            avg_loss = running_loss / (batch_idx + 1)
            swanlab.log({
                "train/loss": avg_loss,
                "train/batch": batch_idx,
                "train/epoch_progress": progress
            }, step=epoch * total_batches + batch_idx)
            
            print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}% "
                  f"Loss: {avg_loss:.4f}", end="")
    
    # 记录更多训练指标
    epoch_loss = running_loss / total_batches
    epoch_time = time() - start_time
    samples_per_second = len(train_dataset) / epoch_time
    
    metrics = {
        "train/epoch_loss": epoch_loss,
        "train/epoch_time": epoch_time,
        "train/samples_per_second": samples_per_second,
        "train/learning_rate": optimizer.param_groups[0]["lr"]
    }
    swanlab.log(metrics, step=epoch)
    
    print(f"\nEpoch {epoch + 1} Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

def test(epoch):
    start_time = time()
    correct = 0
    total = 0
    total_loss = 0
    total_batches = len(test_loader)
    
    print("\nEvaluating on test set...")
    model.eval()  # 设置为评估模式
    
    with no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # 计算进度百分比
            progress = (batch_idx + 1) / total_batches * 100
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predicted = mymax().forward(outputs.data, axis=1)
            predicted = np.round(predicted)

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

def save_checkpoint(model, optimizer, epoch, accuracy, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, checkpoint)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    if os.path.exists(filename + '.npy'):
        checkpoint = np.load(filename + '.npy', allow_pickle=True).item()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['accuracy']
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {start_epoch} with accuracy {best_accuracy:.2f}%")
        return start_epoch, best_accuracy, best_loss
    return 0, 0.0, float('inf')

class LRScheduler:
    def __init__(self, optimizer, patience=3, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        self.best_loss = None
        self.bad_epochs = 0
        
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.bad_epochs = 0
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    if new_lr != old_lr:
                        print(f"\nReducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
        else:
            self.best_loss = val_loss
            self.bad_epochs = 0

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

if __name__ == '__main__':
    total_params = sum(p.array().size for p in model.parameters())
    print(f"\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Device: {run.config.device}")
    
    # 设置检查点路径
    checkpoint_dir = 'checkpoints/mnist_lenet'
    best_model_path = os.path.join(checkpoint_dir, 'best_model')
    latest_model_path = os.path.join(checkpoint_dir, 'latest_model')
    
    # 尝试加载之前的检查点
    start_epoch, best_accuracy, best_loss = load_checkpoint(model, optimizer, latest_model_path)
    
    # 初始化学习率调度器和早停
    lr_scheduler = LRScheduler(optimizer, patience=3, min_lr=1e-6, factor=0.1)
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)
    
    print(f"\nStarting training from epoch {start_epoch + 1} for {run.config.num_epochs} epochs...")
    
    try:
        for epoch in range(start_epoch, run.config.num_epochs):
            train(epoch)
            test(epoch)
            
            # 获取当前epoch的指标
            current_accuracy = swanlab.get_latest_value("test/accuracy")
            current_loss = swanlab.get_latest_value("test/loss")
            
            # 更新学习率调度器和早停
            lr_scheduler.step(current_loss)
            if early_stopping(current_loss):
                print("\nEarly stopping triggered")
                break
            
            # 保存最新的检查点
            save_checkpoint(model, optimizer, epoch + 1, current_accuracy, current_loss, latest_model_path)
            
            # 如果是最佳模型，保存最佳检查点
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                save_checkpoint(model, optimizer, epoch + 1, current_accuracy, current_loss, best_model_path)
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
            
            # 每个epoch后记录学习率
            swanlab.log({
                "train/learning_rate": optimizer.param_groups[0]["lr"]
            }, step=epoch)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(model, optimizer, epoch + 1, current_accuracy, current_loss, latest_model_path)
        print("Checkpoint saved. Exiting...")