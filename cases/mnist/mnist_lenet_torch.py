import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from time import time
import swanlab

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层: 1个输入通道, 6个输出通道, 5x5的卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第一个池化层: 2x2
        self.pool1 = nn.MaxPool2d(2)
        # 第二个卷积层: 6个输入通道, 16个输出通道, 5x5的卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二个池化层: 2x2
        self.pool2 = nn.MaxPool2d(2)
        # 三个全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, criterion, epoch, run):
    model.train()
    start_time = time()
    running_loss = 0.0
    total_batches = len(train_loader)
    print(f"\n【Epoch {epoch + 1}】")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 每10个batch更新一次损失
        if batch_idx % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            run.log({
                "main/loss": avg_loss,
            }, step=epoch * total_batches + batch_idx)
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / target.size(0)
            
            # 打印进度条
            print(f"Batch [{batch_idx + 1}/{total_batches}]  - Loss: [{avg_loss:.4f}]  - Accuracy: [{accuracy:.2f}%]", end="\r")
    
    # 打印epoch总结
    epoch_loss = running_loss / total_batches
    epoch_time = time() - start_time
    print(f"\nTime Used: {epoch_time:.2f} s")

    run.log({
        "train/epoch_loss": epoch_loss,
        "train/epoch_time": epoch_time,
        "main/samples_per_second": len(train_loader.dataset) / epoch_time
    }, step=epoch)

def test(model, device, test_loader, criterion, epoch, run):
    model.eval()
    test_loss = 0
    correct = 0
    total_batches = len(test_loader)
    start_time = time()
    
    print(f"\n【Test {epoch + 1}】")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 打印进度条
            accuracy = 100. * correct / ((batch_idx + 1) * target.size(0))
            print(f"Batch [{batch_idx + 1}/{total_batches}]  - Accuracy: [{accuracy:.2f}%]", end="\r")

    # 计算最终指标
    test_loss /= total_batches
    accuracy = 100. * correct / len(test_loader.dataset)
    test_time = time() - start_time
    
    # 记录到SwanLab
    run.log({
        "main/accuracy": accuracy,
        "test/loss": test_loss,
        "test/time": test_time
    }, step=epoch)
    
    # 打印测试结果总结
    print(f"\nTime Used: {test_time:.2f} s")

def main():
    # 初始化 SwanLab
    run = swanlab.init(
        project="MNIST-LeNet-1230",
        experiment_name="MNIST-LeNet-Torch",
        config={
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "batch_size": 64,
            "num_epochs": 10,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    )
    
    # 设置设备
    device = torch.device(run.config.device)
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, 'data')
    mnist_dir = os.path.join(data_dir, 'mnist')
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(mnist_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(mnist_dir, train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=run.config.batch_size)
    
    # 创建模型
    model = LeNet().to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Device: {device}")
    print(f"\nStarting training for {run.config.num_epochs} epochs...")
    
    # 设置优化器和损失函数
    optimizer = Adam(model.parameters(), lr=run.config.learning_rate)
    criterion = nn.NLLLoss()
    
    # 训练循环
    for epoch in range(run.config.num_epochs):
        train(model, device, train_loader, optimizer, criterion, epoch, run)
        test(model, device, test_loader, criterion, epoch, run)
        
        # 每个epoch后记录学习率
        swanlab.log({
            "train/learning_rate": optimizer.param_groups[0]["lr"]
        }, step=epoch)

if __name__ == '__main__':
    main() 