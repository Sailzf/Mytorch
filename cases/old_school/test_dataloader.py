from torchvision import transforms, datasets
import numpy as np
from mytorch.tensor import Tensor
from mytorch.dataset import MNISTDataset
from mytorch.dataloader import DataLoader

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

def test_mnist_loader():
    # 设置参数
    batch_size = 32
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 只加载训练集进行测试
    mnist_train = datasets.MNIST(root='data/mnist/', train=True, 
                               download=True, transform=transform)
    
    # 准备数据集
    train_dataset = prepare_mnist_data(mnist_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 测试数据加载
    print(f"数据集大小: {len(train_dataset)}")
    print(f"批次数量: {len(train_loader)}")
    
    # 获取第一个批次进行检查
    first_batch = next(iter(train_loader))
    images, labels = first_batch
    
    print("\n第一个批次的信息:")
    print(f"图像形状: {images.shape}")  # 应该是 (batch_size, 1, 28, 28)
    print(f"标签形状: {labels.shape}")  # 应该是 (batch_size,)
    
    # 测试迭代是否正常
    print("\n测试数据迭代:")
    for i, (images, labels) in enumerate(train_loader):
        if i < 3:  # 只打印前3个批次的信息
            print(f"批次 {i+1}: 图像形状 {images.shape}, 标签形状 {labels.shape}")
        elif i == 3:
            print("...")
            break

if __name__ == '__main__':
    test_mnist_loader() 