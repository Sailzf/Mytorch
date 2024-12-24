import cupy as cp
from .tensor import Tensor
from .module import Linear, Conv2D

# 测试 Linear 初始化
def test_linear():
    device = "cuda"
    in_features = 4
    out_features = 2

    print("Testing Linear Initialization")
    linear = Linear(in_features=in_features, out_features=out_features, device=device)

    print("Linear Weight Shape:", linear.weight.shape)
    print("Linear Weight Device:", linear.weight.device)
    if linear.bias is not None:
        print("Linear Bias Shape:", linear.bias.shape)
        print("Linear Bias Device:", linear.bias.device)
    print()

# 测试 Conv2D 初始化
def test_conv2d():
    device = "cuda"
    in_channels = 1
    out_channels = 2
    kernel_size = (3, 3)

    print("Testing Conv2D Initialization")
    conv = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, device=device)

    print("Conv2D Weight Shape:", conv.weight.shape)
    print("Conv2D Weight Device:", conv.weight.device)
    print("Conv2D Bias Shape:", conv.bias.shape)
    print("Conv2D Bias Device:", conv.bias.device)
    print()

# 主函数
if __name__ == "__main__":
    test_linear()
    test_conv2d()
