import numpy as np
import cupy as cp
from .tensor import Tensor
from .module import Module
from  import init, cuda
from .functions import cross_entropy
from .loss import CrossEntropyLoss, NLLLoss, MSELoss

from .module import Conv2D, MaxPooling2D
from .module import Linear
import .functions as F
from time import time

from .cuda import (
    Device,
    get_device_from_array,
    CpuDevice,
    GpuDevice,
    check_cuda_available,
    get_device,
    get_array_module,
    gpu_available
)

class SimpleCNN(Module):
    """
    一个简单的卷积神经网络：
    Conv2D -> MaxPooling2D -> Flatten -> Fully Connected
    """
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=1, padding=0)
        self.pool = MaxPooling2D(pool_h=2, pool_w=2, stride=2)
        self.fc = Linear(in_features=1 * 1 * 1, out_features=1)  # 最终输出一个标量

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)  # 卷积层
        x = F.relu(x)
        x = self.pool(x)   # 最大池化层
        x = x.reshape(x.shape[0], -1)  # 展平
        x = self.fc(x)     # 全连接层
        return x
    
    # def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)  # 卷积层
        # print("After Conv1:", x.shape)
        x = F.relu(x)
        # print("After relu:", x.shape)

        x = self.pool(x)   # 最大池化层
        # print("After Pooling:", x.shape)
        x = x.reshape(x.shape[0], -1)  # 展平
        # print("After Flatten:", x.shape)
        x = self.fc(x)     # 全连接层
        # print("After FC:", x.shape)
        return x

def numpy_test():
    # 固定输入 (1, 1, 4, 4): 形状为 (batch_size, channels, height, width)
    x = Tensor(np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32), requires_grad=True)

    # 固定目标输出为 1.0
    target = Tensor(np.array([1.0], dtype=np.float32))
    model = SimpleCNN()
    test_cnn(x, target, model)


def cupy_test():
    x = Tensor(cp.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=cp.float32), requires_grad=True)

    # 固定目标输出为 1.0
    target = Tensor(cp.array([1.0], dtype=cp.float32))
    model = SimpleCNN()
    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
    model.to(device)
    test_cnn(x, target, model)


# 测试函数
def test_cnn(x, target, model):
    start_time = time()  # 记录训练开始时间
    # 初始化网络

    # 定义简单的均方误差损失函数
    # def mse_loss(output, target):
    #     return ((output - target) ** 2).mean()

    # 定义学习率
    learning_rate = 0.01

    criterion = MSELoss()

    # loss = criterion(output, target)  # 使用交叉熵损失

    # 训练若干轮
    for epoch in range(200):
        # 前向传播
        output = model(x)
        
        # 计算损失
        loss = criterion(output, target)

        # 打印损失值
        if epoch % 49 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

        # 反向传播
        loss.backward()

        # 更新权重
        for param in model.parameters():
            param.data -= learning_rate * param.grad

        # 清零梯度
        for param in model.parameters():
            param.zero_grad()

    # 最终结果
    print("最终输出：", model(x).item())
    end_time = time()
    print(f"training time: {end_time - start_time:.2f} seconds")
    
# 运行测试
if __name__ == "__main__":
    numpy_test()
    cupy_test()