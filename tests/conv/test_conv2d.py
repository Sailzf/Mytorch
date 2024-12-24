import numpy as np
from .tensor import Tensor

import .functions as F
from .tensor import Tensor, debug_mode, cuda

from .module import Conv2d, MaxPool2d
from .module import Conv2D, MaxPooling2D
# 定义输入张量
import numpy as np


#11/19
# 定义测试函数
def test_conv2D():
    # 初始化输入张量（形状: N, C, H, W）
    input_data = Tensor(np.array([[[[1, 2, 3],
                                     [4, 5, 6],
                                     [7, 8, 9]]]]))  # 单通道 3x3 输入

    # 定义卷积核（形状: out_channels, in_channels, kernel_height, kernel_width）
    kernel = Tensor(np.array([[[[1, 0],
                                 [0, -1]]]]))  # 单通道 2x2 滤波器

    # 定义偏置
    bias = Tensor(np.array([0]))  # 偏置为0

    # 创建 Conv2D 实例
    conv2d_layer = Conv2D(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=1, padding=0)

    # 手动设置权重和偏置
    conv2d_layer.weight.data = kernel.data
    conv2d_layer.bias.data = bias.data

    # 执行前向传播
    output = conv2d_layer.forward(input_data)
    # 预期输出
    expected_output = np.array([[[[-4, -4],
                                   [-4, -4]]]])  # 通过手动卷积计算得到的结果

    # 结果对比
    if np.allclose(output.data, expected_output):
        print("Conv2D 测试通过！")
        print("输出:", output.data)
    else:
        print("Conv2D 测试失败！")
        print("输出:", output.data)
        print("预期:", expected_output)

# 运行测试
test_conv2D()

print("\n\nConv2d 开始测试")

# 定义一个简单的 Conv2d
conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
conv.weight.data = np.array([[[[1, 0], [0, -1]]]])  # 1x1x2x2 卷积核
conv.bias.data = np.array([0])  # 无偏置

# 定义一个简单的 MaxPool2d
pool = MaxPool2d(kernel_size=2, stride=1)

# 输入张量（1x1x4x4）
input_tensor = Tensor(np.array([[[[1, 2, 3, 0],
                                  [4, 5, 6, 1],
                                  [7, 8, 9, 2],
                                  [3, 0, 1, 4]]]]))

# 计算 Conv2d 的输出
conv_output = conv(input_tensor)

# 计算 MaxPool2d 的输出
pooled_output = pool(conv_output)

# 打印结果
print("输入张量:")
print(input_tensor.data)

print("\nConv2d 输出:")
print(conv_output.data)

print("\nMaxPool2d 输出:")
print(pooled_output.data)




#     # # 定义梯度 (输出的形状和 forward 的输出一致)
#     # dout = Tensor(np.array([[[[1, 1],
#     #                            [1, 1]]]]))  # 假设损失对输出的梯度为全 1

#     # ############ 反向传播
#     # dx = conv2d_layer.backward(dout)

#     # # 手动计算梯度
#     # # 对权重的梯度
#     # dW_manual = np.array([[[[12, 16],
#     #                         [24, 28]]]])  # 手工计算权重梯度
#     # # 对输入的梯度
#     # dx_manual = np.array([[[[1, 1, 0],
#     #                          [1, 1, 0],
#     #                          [0, 0, 0]]]])  # 手工计算输入梯度

#     # # 对比结果
#     # print("权重梯度比较:")
#     # print("自动计算:\n", conv2d_layer.weight.grad)
#     # print("手工计算:\n", dW_manual)
#     # print("是否匹配:", np.allclose(conv2d_layer.weight.grad, dW_manual))

#     # print("\n输入梯度比较:")
#     # print("自动计算:\n", dx.data)
#     # print("手工计算:\n", dx_manual)
#     # print("是否匹配:", np.allclose(dx.data, dx_manual))








# device = cuda.get_device("cuda:0" if cuda.is_available() else "cpu")
# print(f"current device:{device}")


# import cupy as cp

# # 输入张量和参数
# x = cp.random.randn(10, 3, 32, 32).astype(np.float32)  # 输入 (batch_size, in_channels, height, width)
# weight = cp.random.randn(16, 3, 3, 3).astype(np.float32)  # 卷积核 (out_channels, in_channels, kernel_h, kernel_w)
# bias = cp.random.randn(16).astype(np.float32)  # 偏置

# # 初始化 Conv2d 模块
# in_channels = 3
# out_channels = 16
# kernel_size = (3, 3)  # 卷积核大小
# stride = 1
# padding = 1

# conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)

# # 进行前向传播
# output = conv.forward(x)
# print("Output shape:", output.shape)

# # 反向传播
# grad_output = cp.random.randn(*output.shape).astype(np.float32)
# grad_x, grad_weight, grad_bias = conv.backward(grad_output)

# print("Output shape:", output.shape)
# print("Gradients calculated successfully.")


# def test_conv2d():
#     # 设置输入和卷积核的尺寸
#     batch_size, in_channels, height, width = 1, 3, 5, 5
#     out_channels, kernel_height, kernel_width = 2, 3, 3

#     # 使用随机数初始化输入和卷积核
#     x = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
#     weight = np.random.randn(out_channels, in_channels, kernel_height, kernel_width).astype(np.float32)
#     bias = np.random.randn(out_channels).astype(np.float32)

#     with debug_mode():
#         # 使用  创建张量并计算卷积
#         mx = Tensor(x, requires_grad=True, device=device)
#         mw = Tensor(weight, requires_grad=True, device=device)
#         mb = Tensor(bias, requires_grad=True, device=device)
#         my = F.Conv2d(mx, mw, mb)

#         # 使用 PyTorch 创建张量并计算卷积
#         tx = torch.tensor(x, requires_grad=True)
#         tw = torch.tensor(weight, requires_grad=True)
#         tb = torch.tensor(bias, requires_grad=True)
#         ty = torch.nn.functional.conv2d(tx, tw, tb)

#         # 验证前向输出是否一致
#         assert np.allclose(my.data, ty.data), f"Output mismatch:  vs torch"

#         # 反向传播
#         my.sum().backward()
#         ty.sum().backward()

#         # 验证反向传播计算的梯度是否一致
#         assert np.allclose(mx.grad, tx.grad), f"Gradient mismatch in input:  vs torch"
#         assert np.allclose(mw.grad, tw.grad), f"Gradient mismatch in weight:  vs torch"
#         assert np.allclose(mb.grad, tb.grad), f"Gradient mismatch in bias:  vs torch"


# def test_conv2d_batch():
#     # 设置输入和卷积核的尺寸（批量大小）
#     batch_size, in_channels, height, width = 4, 3, 5, 5
#     out_channels, kernel_height, kernel_width = 2, 3, 3

#     # 使用随机数初始化输入和卷积核
#     x = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
#     weight = np.random.randn(out_channels, in_channels, kernel_height, kernel_width).astype(np.float32)
#     bias = np.random.randn(out_channels).astype(np.float32)

#     with debug_mode():
#         # 使用  创建张量并计算卷积
#         mx = Tensor(x, requires_grad=True, device=device)
#         mw = Tensor(weight, requires_grad=True, device=device)
#         mb = Tensor(bias, requires_grad=True, device=device)
#         my = F.conv2d(mx, mw, mb)

#         # 使用 PyTorch 创建张量并计算卷积
#         tx = torch.tensor(x, requires_grad=True)
#         tw = torch.tensor(weight, requires_grad=True)
#         tb = torch.tensor(bias, requires_grad=True)
#         ty = torch.nn.functional.conv2d(tx, tw, tb)

#         # 验证前向输出是否一致
#         assert np.allclose(my.data, ty.data), f"Output mismatch:  vs torch"

#         # 反向传播
#         my.sum().backward()
#         ty.sum().backward()

#         # 验证反向传播计算的梯度是否一致
#         assert np.allclose(mx.grad, tx.grad), f"Gradient mismatch in input:  vs torch"
#         assert np.allclose(mw.grad, tw.grad), f"Gradient mismatch in weight:  vs torch"
#         assert np.allclose(mb.grad, tb.grad), f"Gradient mismatch in bias:  vs torch"





# # 运行测试
# if __name__ == "__main__":
#     test_conv2d()
#     # test_conv2d_batch()
#     print("All tests passed!")