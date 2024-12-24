from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import numpy as np

from mytorch.tensor import no_grad
from mytorch.tensor import Tensor as MetaTensor
import mytorch.module as nn  # 网络模块
from mytorch.module import Module, Linear, Conv2D, MaxPooling2D
import mytorch.functions as F  # 激活函数及其他运算
from mytorch.functions import relu, softmax
from mytorch.optim import Adam, SGD, Adagrad  # 优化器
from mytorch.loss import CrossEntropyLoss, NLLLoss  # 损失函数
from mytorch import cuda
import argparse
from mytorch.ops import Max as mymax

from time import time
from mytorch.distributed import RingAllReduce


class Feedforward(nn.Module):

    def __init__(self):
        super(Feedforward, self).__init__()
        self.linear = Linear(784, 10)

    def forward(self, x):
        return F.log_softmax(self.linear(x))


def torch_to_metatensor(torch_tensor, requires_grad=False):
    data = np.array(torch_tensor.data)
    dtype = torch_tensor.dtype
    return MetaTensor(data=data, requires_grad=requires_grad, dtype=dtype)


def partition_dataset(dataset, world_size, rank):
    """按进程数量划分数据集"""
    total_size = len(dataset)
    size_per_rank = total_size // world_size  #考虑让最后一个主机多处理一些数据集
    if rank == world_size - 1:
        indices = list(range(rank * size_per_rank, total_size))
    else:
        indices = list(range(rank * size_per_rank, (rank + 1) * size_per_rank))
    return torch.utils.data.Subset(dataset, indices)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    args = parser.parse_args()

    # 分布式训练配置
    nodes = [
        ('113.54.255.136', 29500),  # 进程0
        ('113.54.253.207', 29501),  # 进程1
        ('113.54.253.207', 29502),  # 进程2
        # 根据实际需要添加更多节点
    ]

    # 初始化分布式环境
    distributed = RingAllReduce(args.rank, args.world_size, nodes)

    # 数据加载和预处理
    batch_size = 64
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # 加载并划分训练数据
    train_dataset = datasets.MNIST(root='data/mnist/',
                                   train=True,
                                   download=True,
                                   transform=transform)
    train_subset = partition_dataset(train_dataset, args.world_size, args.rank)
    train_loader = DataLoader(train_subset,
                              batch_size=batch_size,
                              shuffle=True)

    # 加载测试数据（所有进程都使用完整测试集）
    test_dataset = datasets.MNIST(root='data/mnist',
                                  train=False,
                                  download=True,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    # 初始化模型、损失函数和优化器
    model = Feedforward()
    criterion = NLLLoss()
    optimizer = SGD(model.parameters(), lr=0.03)

    def train(epoch):
        running_loss = 0.0
        for batch_idx, (inputs, target) in enumerate(train_loader):
            # 数据转换
            inputs = torch_to_metatensor(inputs)
            target = torch_to_metatensor(target)
            inputs = inputs.view(-1, 784)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            print(f"梯度同步前")
            start_time = time()  # 记录训练开始时间

            # 同步梯度
            for param in model.parameters():
                if param.grad is not None:
                    # 使用Ring-AllReduce同步梯度
                    synced_grad = distributed.allreduce(param.grad)
                    #print(f"训练了一个批次")#这里没有出来
                    # 因为是累加，所以需要除以进程数
                    param.grad.data = synced_grad.data / args.world_size
            end_time = time()
            print(f"梯度同步后")
            print(f"梯度同步时间: {end_time - start_time:.2f} seconds")

            # 更新参数
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(
                    f'[Rank {args.rank}, Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}'
                )
                running_loss = 0.0

    def test():
        correct = 0
        total = 0
        with no_grad():
            for inputs, labels in test_loader:
                inputs = torch_to_metatensor(inputs)
                labels = torch_to_metatensor(labels)
                inputs = inputs.view(-1, 784)

                outputs = model(inputs)
                outputs = MetaTensor(outputs.data)

                predicted = np.argmax(outputs.data, axis=1)
                labels = labels.array()

                total += labels.size
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Rank {args.rank} Accuracy on test set: {accuracy:.2f}%')

    # 训练循环
    try:
        for epoch in range(15):
            train(epoch)
            if args.rank == 0:  # 只在主进程上测试
                test()
    except Exception as e:
        print(f"Rank {args.rank} encountered error: {str(e)}")
    finally:
        print(f"Rank {args.rank} training completed")


if __name__ == '__main__':
    main()
