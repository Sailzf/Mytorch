import numpy as np
import cupy as cp
from .loss import NLLLoss
from .module import Module, Linear
from .optim import SGD
from .tensor import Tensor
import .functions as F
from  import cuda
from time import time

class SoftmaxRegression(Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # 计算对数概率
        return F.log_softmax(self.linear(x))

def numpy_test():

    # 构造简单的 cupy 输入
    # 输入是形状为 (5, 4) 的数据，表示 5 个样本，每个样本有 4 个特征
    X = Tensor(np.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 3.0, 2.0, 4.0],
        [4.0, 2.0, 1.0, 3.0],
        [2.0, 1.0, 4.0, 3.0],
    ], dtype=np.float32))

    # 目标是形状为 (5, 3) 的 one-hot 标签，表示 3 类分类
    y = Tensor(np.array([
        [1, 0, 0],  # 类别 0
        [0, 1, 0],  # 类别 1
        [0, 0, 1],  # 类别 2
        [1, 0, 0],  # 类别 0
        [0, 1, 0],  # 类别 1
    ], dtype=np.float32))

    # 定义模型、优化器和损失函数
    model = SoftmaxRegression(4, 3)  # 输入维度 4，输出维度 3
    device = cuda.get_device("cpu")
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.1)  # 学习率 0.1
    loss_fn = NLLLoss()  # 负对数似然损失
    start_time = time()  # 记录训练开始时间

    # 训练循环
    epochs = 100
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)

        # 计算损失
        loss = loss_fn(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印损失值
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # 测试模型的输出
    print("Final outputs:")
    print(model(X).array())  # 打印最终的输出
    end_time = time()
    print(f"training time: {end_time - start_time:.2f} seconds")    

def cupy_test():

    # 构造简单的 cupy 输入
    # 输入是形状为 (5, 4) 的数据，表示 5 个样本，每个样本有 4 个特征
    X = Tensor(cp.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, 3.0, 2.0, 4.0],
        [4.0, 2.0, 1.0, 3.0],
        [2.0, 1.0, 4.0, 3.0],
    ], dtype=cp.float32))

    # 目标是形状为 (5, 3) 的 one-hot 标签，表示 3 类分类
    y = Tensor(cp.array([
        [1, 0, 0],  # 类别 0
        [0, 1, 0],  # 类别 1
        [0, 0, 1],  # 类别 2
        [1, 0, 0],  # 类别 0
        [0, 1, 0],  # 类别 1
    ], dtype=cp.float32))

    # 定义模型、优化器和损失函数
    model = SoftmaxRegression(4, 3)  # 输入维度 4，输出维度 3
    device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.1)  # 学习率 0.1
    loss_fn = NLLLoss()  # 负对数似然损失
    start_time = time()  # 记录训练开始时间

    # 训练循环
    epochs = 100
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)

        # 计算损失
        loss = loss_fn(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印损失值
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # 测试模型的输出
    print("Final outputs:")
    print(model(X).array())  # 打印最终的输出
    end_time = time()
    print(f"training time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    numpy_test()
    cupy_test()