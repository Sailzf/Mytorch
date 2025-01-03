import os
import sys

# 获取项目根目录并添加到 Python 路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import cupy as cp
import numpy as np
from time import time
import swanlab

from mytorch.ops import Max as mymax
from mytorch.tensor import Tensor, no_grad
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader, prepare_mnist_data
import mytorch.module as nn
from mytorch.module import Module, Linear, Conv2D, MaxPooling2D
import mytorch.functions as F
from mytorch.functions import relu, softmax
from mytorch.optim import Adam, Adagrad
from mytorch.loss import CrossEntropyLoss, NLLLoss
from mytorch import cuda
import argparse
from mytorch.distributed import RingAllReduce


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


def partition_dataset(dataset, world_size, rank):
    """将数据集划分为多个部分用于分布式训练
    
    Args:
        dataset: 要划分的数据集
        world_size (int): 总的进程数量
        rank (int): 当前进程的序号（从0开始）
        
    Returns:
        Dataset: 划分后的数据集子集
    """
    # 计算每个进程应该分到的数据量
    total_size = len(dataset)
    base_size = total_size // world_size
    remainder = total_size % world_size
    
    # 如果有余数，最后一个进程会多分到一些数据
    if rank == world_size - 1:
        partition_size = base_size + remainder
    else:
        partition_size = base_size
    
    # 计算当前进程的数据起始索引
    start_idx = rank * base_size
    end_idx = start_idx + partition_size
    
    # 创建新的数据集
    if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
        # 对于MNIST等标准数据集
        dataset.data = dataset.data[start_idx:end_idx]
        dataset.targets = dataset.targets[start_idx:end_idx]
    else:
        # 对于自定义数据集，可能需要其他处理方式
        raise NotImplementedError("当前只支持包含data和targets属性的数据集")
    
    return dataset

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    args = parser.parse_args()

    # 定义基本配置参数
    config = {
        "optimizer": "Adam",
        "learning_rate": 0.03,
        "batch_size": 64,
        "num_epochs": 10,
        "device": "cuda" if cuda.is_available() else "cpu",
        "world_size": args.world_size
    }

    # 只在主进程(rank 0)初始化 SwanLab
    if args.rank == 0:
        run = swanlab.init(
            project="MNIST-LeNet-Distributed",
            experiment_name="MNIST-LeNet-cupy-distributed",
            config=config
        )

    # 分布式训练配置
    nodes = [
        ('127.0.0.1', 29500),  # 进程0
        ('127.0.0.1', 29501),  # 进程1
        ('127.0.0.1', 29502),  # 进程2
    ]



    # 初始化分布式环境
    distributed = RingAllReduce(args.rank, args.world_size, nodes)

    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, 'data')

    # 加载和准备数据
    train_dataset = prepare_mnist_data(root=data_dir, backend='cupy', train=True)
    # 划分训练数据集
    train_dataset = partition_dataset(train_dataset, args.world_size, args.rank)
    
    test_dataset = prepare_mnist_data(root=data_dir, backend='cupy', train=False)

    # 修改 DataLoader 的 batch_size 参数类型
    batch_size = int(config["batch_size"])  # 确保 batch_size 是整数
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = LeNet()
    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
    model.to(device)

    criterion = NLLLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    def train(epoch):
        running_loss = 0.0
        total_batches = len(train_loader)
        print(f"\n【Epoch {epoch + 1}】")
        
        # 用于计算整个 epoch 的统计信息
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (inputs, target) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            # 同步梯度
            for param in model.parameters():
                if param.grad is not None:
                    # 将 param.grad (cupy.ndarray) 转换为 MetaTensor
                    grad_tensor = Tensor(param.grad)  # 使用 Tensor 类包装 cupy.ndarray
                    synced_grad = distributed.allreduce(grad_tensor)
                    # 使用 copyto 来更新数据
                    cp.copyto(param.grad, synced_grad.data / args.world_size)

            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            # 计算准确率
            predicted = mymax().forward(outputs.data, axis=1)
            predicted = cp.rint(predicted)
            correct = (predicted == target.array()).sum().item()
            accuracy = 100 * correct / target.array().size
            
            epoch_correct += correct
            epoch_total += target.array().size

            # 只在主进程中记录日志
            if args.rank == 0 and batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                # 只在主进程中使用 swanlab.log
                swanlab.log({
                    "main/loss": avg_loss,
                }, step=epoch * total_batches + batch_idx)
                
                print(f"Batch [{batch_idx + 1}/{total_batches}] - Loss: [{avg_loss:.4f}] - Accuracy: [{accuracy:.2f}%]", end="\r")
            else:
                # 非主进程只打印进度
                print(f"Rank {args.rank} - Batch [{batch_idx + 1}/{total_batches}] - Loss: [{loss.item():.4f}] - Accuracy: [{accuracy:.2f}%]", end="\r")

        # 在主进程中记录训练指标
        if args.rank == 0:
            epoch_avg_loss = epoch_loss / total_batches
            epoch_accuracy = 100 * epoch_correct / epoch_total
            run.log({
                "train_loss": epoch_avg_loss,
                "train_accuracy": epoch_accuracy
            })

    def test():
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        total_batches = len(test_loader)
        print(f"\n【Test {epoch + 1}】")
        with no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                predicted = mymax().forward(outputs.data, axis=1)
                predicted = cp.rint(predicted)

                total_loss += loss.item()
                total += labels.array().size
                correct += (predicted == labels.array()).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / total_batches
        print(f'Rank {args.rank} Accuracy on test set: {accuracy:.2f}%')

        # 在主进程中记录测试指标
        if args.rank == 0:
            run.log({
                "test_loss": avg_loss,
                "test_accuracy": accuracy
            })

    # 训练循环
    try:
        for epoch in range(10):
            train(epoch)
            if args.rank == 0:  # 只在主进程上测试
                test()
    except Exception as e:
        print(f"Rank {args.rank} encountered error: {str(e)}")
    finally:
        print(f"Rank {args.rank} training completed")
        if args.rank == 0:
            run.finish()

if __name__ == '__main__':
    main()
