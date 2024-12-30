# -*- coding: utf-8 -*-
"""
LeNet-5 MNIST 分类器
实现了经典的LeNet-5架构，用于MNIST手写数字分类
包含数据缓存机制和详细的训练过程监控
"""

import os
import numpy as np
from time import time
from torchvision import transforms, datasets

# mytorch imports
from mytorch.ops import Max as mymax
from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader
import mytorch.module as nn
from mytorch.module import Module, Linear, Conv2D, MaxPooling2D
import mytorch.functions as F
from mytorch.optim import Adam
from mytorch.loss import NLLLoss
from mytorch import cuda

# 实验跟踪
import swanlab

class LeNet(nn.Module):
    """
    LeNet-5 卷积神经网络架构
    输入: 1x28x28 的图像
    输出: 10个类别的概率分布
    """
    def __init__(self):
        super(LeNet, self).__init__()
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            # 第一个卷积块
            Conv2D(1, 6, (5, 5)),  # 输入1通道，输出6通道，5x5卷积核
            MaxPooling2D(2, 2, 2),  # 2x2池化，步长2
            
            # 第二个卷积块
            Conv2D(6, 16, (5, 5)),  # 输入6通道，输出16通道，5x5卷积核
            MaxPooling2D(2, 2, 2)   # 2x2池化，步长2
        )
        
        # 分类器层
        self.classifier = nn.Sequential(
            Linear(16 * 4 * 4, 120),  # 展平后的特征图到120
            Linear(120, 84),          # 120到84
            Linear(84, 10)            # 84到输出类别
        )

    def forward(self, x):
        # 特征提取
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)  # 展平特征图
        
        # 分类
        x = self.classifier(x)
        return F.log_softmax(x)

class MNISTTrainer:
    """训练器类，封装了训练和评估逻辑"""
    def __init__(self, config):
        self.config = config
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        """准备数据集和数据加载器"""
        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 设置缓存目录
        cache_dir = 'data/mnist/processed'
        train_cache = os.path.join(cache_dir, 'train')
        test_cache = os.path.join(cache_dir, 'test')
        
        # 加载数据集
        mnist_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
        
        # 准备数据集
        self.train_dataset = self._prepare_dataset(mnist_train, train_cache)
        self.test_dataset = self._prepare_dataset(mnist_test, test_cache)
        
        # 创建数据加载器
        self.train_loader = DataLoader(self.train_dataset, 
                                     batch_size=self.config.batch_size, 
                                     shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, 
                                    batch_size=self.config.batch_size, 
                                    shuffle=False)
    
    def _prepare_dataset(self, dataset, cache_file):
        """准备数据集，支持缓存机制"""
        if os.path.exists(cache_file + '_data.npy'):
            print(f"Loading cached dataset from {cache_file}")
            data = np.load(cache_file + '_data.npy')
            targets = np.load(cache_file + '_targets.npy')
        else:
            print("Converting dataset and creating cache...")
            data, targets = [], []
            for x, y in dataset:
                data.append(np.array(x))
                targets.append(y)
            
            data = np.stack(data)
            targets = np.array(targets)
            
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file + '_data.npy', data)
            np.save(cache_file + '_targets.npy', targets)
            
        return MNISTDataset(Tensor(data, requires_grad=False),
                           Tensor(targets, requires_grad=False))
    
    def setup_model(self):
        """初始化模型、损失函数和优化器"""
        self.model = LeNet()
        self.criterion = NLLLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        start_time = time()
        running_loss = 0.0
        total_batches = len(self.train_loader)
        
        print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
        print(f"Training on {len(self.train_dataset)} samples with batch size {self.config.batch_size}")
        
        self.model.train()
        for batch_idx, (inputs, target) in enumerate(self.train_loader):
            # 训练步骤
            outputs = self.model(inputs)
            loss = self.criterion(outputs, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新统计
            running_loss += loss.item()
            progress = (batch_idx + 1) / total_batches * 100
            
            # 记录和显示
            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                self._log_training_progress(epoch, batch_idx, total_batches, 
                                         progress, avg_loss)
        
        # 记录epoch统计
        self._log_epoch_stats(epoch, running_loss, total_batches, start_time)
    
    def evaluate(self, epoch):
        """评估模型"""
        start_time = time()
        correct = total = total_loss = 0
        total_batches = len(self.test_loader)
        
        print("\nEvaluating on test set...")
        self.model.eval()
        
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.test_loader):
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                predicted = mymax().forward(outputs.data, axis=1)
                predicted = np.round(predicted)
                
                # 更新统计
                total_loss += loss.item()
                total += labels.array().size
                correct += (predicted == labels.array()).sum().item()
                
                # 显示进度
                progress = (batch_idx + 1) / total_batches * 100
                print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}%", 
                      end="")
        
        # 记录和显示结果
        self._log_evaluation_results(epoch, correct, total, total_loss, 
                                   total_batches, start_time)
    
    def _log_training_progress(self, epoch, batch_idx, total_batches, progress, avg_loss):
        """记录训练进度"""
        swanlab.log({"train/loss": avg_loss}, 
                   step=epoch * total_batches + batch_idx)
        print(f"\rProgress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}% "
              f"Loss: {avg_loss:.4f}", end="")
    
    def _log_epoch_stats(self, epoch, running_loss, total_batches, start_time):
        """记录epoch统计信息"""
        epoch_loss = running_loss / total_batches
        epoch_time = time() - start_time
        samples_per_second = len(self.train_dataset) / epoch_time
        
        # 打印统计
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Time Used: {epoch_time:.2f} seconds")
        print(f"Samples/second: {samples_per_second:.2f}")
        
        # 记录到SwanLab
        swanlab.log({
            "train/epoch_loss": epoch_loss,
            "train/epoch_time": epoch_time,
            "train/samples_per_second": samples_per_second
        }, step=epoch)
    
    def _log_evaluation_results(self, epoch, correct, total, total_loss, 
                              total_batches, start_time):
        """记录评估结果"""
        accuracy = 100 * correct / total
        avg_loss = total_loss / total_batches
        test_time = time() - start_time
        samples_per_second = len(self.test_dataset) / test_time
        
        # 记录到SwanLab
        swanlab.log({
            "test/accuracy": accuracy,
            "test/loss": avg_loss,
            "test/time": test_time
        }, step=epoch)
        
        # 打印结果
        print(f"\nTest Set Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Time Used: {test_time:.2f} seconds")
        print(f"Samples/second: {samples_per_second:.2f}")

def main():
    """主函数"""
    # 配置实验
    run = swanlab.init(
        project="MNIST-LeNet",
        experiment_name="MNIST-LeNet",
        config={
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "batch_size": 64,
            "num_epochs": 10,
            "device": "cuda" if cuda.is_available() else "cpu",
        },
    )
    
    # 创建训练器
    trainer = MNISTTrainer(run.config)
    
    # 打印模型信息
    total_params = sum(p.array().size for p in trainer.model.parameters())
    print(f"\nModel Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Device: {run.config.device}")
    print(f"\nStarting training for {run.config.num_epochs} epochs...")
    
    # 训练循环
    for epoch in range(run.config.num_epochs):
        trainer.train_epoch(epoch)
        trainer.evaluate(epoch)
        
        # 记录学习率
        swanlab.log({
            "train/learning_rate": trainer.optimizer.param_groups[0]["lr"]
        }, step=epoch)

if __name__ == '__main__':
    main()