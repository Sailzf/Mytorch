import numpy as np
from .tensor import Tensor
from .module import Module
from .functions import cross_entropy
from .loss import CrossEntropyLoss, NLLLoss, MSELoss
from .module import Conv2D, MaxPooling2D
from .module import Linear
from .optim import Adam, SGD, Adagrad

import .functions as F

class SimpleCNN(Module):
    """
    一个简单的卷积神经网络：
    Conv2D -> ReLU -> MaxPooling2D -> Conv2D -> ReLU -> MaxPooling2D -> Flatten -> Fully Connected
    """
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = MaxPooling2D(pool_h=2, pool_w=2, stride=2)
        self.conv2 = Conv2D(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = MaxPooling2D(pool_h=2, pool_w=2, stride=2)
        self.fc = Linear(in_features=8 * 7 * 7, out_features=1)  # 假设输入为28x28的图像

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)  # 第一个卷积层
        x = F.relu(x)
        x = self.pool1(x)   # 第一个池化层
        x = self.conv2(x)  # 第二个卷积层
        x = F.relu(x)
        x = self.pool2(x)   # 第二个池化层
        x = x.reshape(x.shape[0], -1)  # 展平
        x = self.fc(x)     # 全连接层
        return x


# 测试函数
def test_cnn():
    # 固定输入 (1, 1, 28, 28): 形状为 (batch_size, channels, height, width)
    x = Tensor(np.random.rand(1, 1, 28, 28).astype(np.float32), requires_grad=True)

    # 固定目标输出为 1.0
    target = Tensor(np.array([1.0], dtype=np.float32))

    # 初始化网络
    model = SimpleCNN()

    # # 定义简单的均方误差损失函数
    # def mse_loss(output, target):
    #     return ((output - target) ** 2).mean()

    # 定义学习率
    learning_rate = 0.001

    criterion = MSELoss()

    optimizer = Adam(model.parameters(), lr=0.001)
    # optimizer = SGD(model.parameters(), lr=0.001)

    # 训练若干轮
    for epoch in range(100):
        # 前向传播
        output = model(x)
        
        # 计算损失
        loss = criterion(output, target)

        # 打印损失值
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

        # # 反向传播
        # loss.backward()

        # # 更新权重
        # for param in model.parameters():
        #     param.data -= learning_rate * param.grad

        # # 清零梯度
        # for param in model.parameters():
        #     param.zero_grad()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 最终结果
    print("最终输出：", model(x).item())

# 运行测试
if __name__ == "__main__":
    test_cnn()
