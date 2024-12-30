from torchvision import transforms, datasets
import cupy as cp
from time import time
import os
import swanlab

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

def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=2, save_path=None):
    """训练模型并返回训练历史"""
    print(f"\n开始训练，训练轮数：{num_epochs}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        start_time = time()
        
        for batch_idx, (inputs, target) in enumerate(train_loader):
            # 计算进度百分比
            progress = (batch_idx + 1) / total_batches * 100
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每10个batch打印一次进度
            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"\rEpoch [{epoch+1}/{num_epochs}] Progress: [{batch_idx:>4d}/{total_batches:>4d}] {progress:>3.0f}% "
                      f"Loss: {avg_loss:.4f}", end="")
        
        # 计算epoch的统计信息
        epoch_loss = running_loss / total_batches
        epoch_time = time() - start_time
        
        # 在测试集上评估
        test_accuracy = evaluate(model, test_loader)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Time Used: {epoch_time:.2f} seconds")
        print(f"Samples/second: {len(train_loader.dataset) / epoch_time:.2f}")
    
    # 如果指定了保存路径，保存模型
    if save_path:
        model.save(save_path)
        print(f"\n模型已保存到 {save_path}")

def evaluate(model, test_loader):
    """评估模型并返回准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = mymax().forward(outputs.data, axis=1)
            predicted = cp.round(predicted)
            total += labels.array().size
            correct += (predicted == labels.array()).sum().item()
    
    return 100 * correct / total

def main():
    # 初始化配置
    config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 2,
        "device": "cuda" if cuda.is_available() else "cpu",
    }
    
    # 加载和准备数据
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
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 创建第一个模型并训练
    print("\n=== 第一阶段：训练初始模型 ===")
    model1 = LeNet()
    device = cuda.get_device(config["device"])
    model1.to(device)
    
    criterion = NLLLoss()
    optimizer = Adam(model1.parameters(), lr=config["learning_rate"])
    
    # 训练第一个模型
    train(model1, train_loader, test_loader, optimizer, criterion, device, 
          num_epochs=config["num_epochs"], save_path='model_stage1.pkl')
    
    # 创建第二个模型，加载第一个模型的参数，并继续训练
    print("\n=== 第二阶段：加载模型并继续训练 ===")
    model2 = LeNet()
    model2.to(device)
    model2.load('model_stage1.pkl')
    
    # 冻结卷积层
    print("\n冻结卷积层...")
    model2.freeze_parameters(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    
    # 使用新的优化器（只优化未冻结的参数）
    optimizer = Adam(model2.parameters(), lr=config["learning_rate"])
    
    # 继续训练
    train(model2, train_loader, test_loader, optimizer, criterion, device, 
          num_epochs=config["num_epochs"])
    
    # 清理保存的模型文件
    if os.path.exists('model_stage1.pkl'):
        os.remove('model_stage1.pkl')
        print("\n清理临时文件完成")

if __name__ == '__main__':
    main() 