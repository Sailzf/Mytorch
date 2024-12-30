# MNIST 分类实验

自制深度学习框架 MyTorch，并支持 CPU 和 CUDA 加速，包含了使用不同网络结构对MNIST数据集进行分类的实验代码。

## 网络结构

项目实现了三种不同的网络结构：

1. LeNet
   - 2个卷积层
   - 2个池化层
   - 3个全连接层

2. AlexNet（修改版）
   - 5个卷积层
   - 3个池化层
   - 3个全连接层
   - Dropout正则化

3. 简单CNN
   - 2个卷积层
   - 2个池化层
   - 1个全连接层

## 特点

- 支持 CPU 和 CUDA 加速
- 数据缓存机制，避免重复数据转换
- 详细的训练监控和性能指标
- NVTX性能分析支持
- SwanLab 实验跟踪

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd Mytorch_distributed+data
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. （可选）安装CUDA支持：
```bash
pip install cupy-cuda12x  # 根据你的CUDA版本选择对应的包
```

## 使用方法

1. 运行LeNet训练：
```bash
python cases/mnist_lenet.py
```

2. 运行AlexNet训练：
```bash
python cases/mnist_alexnet_cupy.py
```

3. 运行PyTorch版本的LeNet：
```bash
python cases/mnist_lenet_torch.py
```

## 配置

所有模型都支持通过 SwanLab 配置以下参数：

- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `num_epochs`: 训练轮数
- `device`: 训练设备 (cpu/cuda)
- `dropout_rate`: Dropout率（仅AlexNet）

## 性能监控

- 训练和测试损失
- 准确率
- 训练时间
- 样本处理速度
- GPU利用率（使用NVTX标记）

## 数据缓存

数据集会被预处理并缓存到以下目录：
- LeNet: `data/mnist/processed/`
- AlexNet: `data/mnist/processed_cupy_alexnet/`

缓存可以大大减少后续运行时的数据加载时间。

## 注意事项

1. 首次运行时会下载MNIST数据集
2. AlexNet版本会将图像上采样到224x224
3. 确保有足够的GPU显存（特别是运行AlexNet时）
4. 可以通过修改配置来调整训练参数

## 依赖

- Python >= 3.7
- PyTorch（用于数据加载）
- CuPy（用于CUDA加速）
- SwanLab（用于实验跟踪）
- NVTX（用于性能分析）