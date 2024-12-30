import numpy as np
import cupy as cp
from .tensor import Tensor
from .cuda import get_array_module
from .cuda import GpuDevice, CpuDevice
from .module import Conv2D, MaxPooling2D
from .cuda import (
    Device,
    get_device_from_array,
    CpuDevice,
    GpuDevice,
    check_cuda_available,
    get_device,
    get_array_module
)

# 定义一个测试函数
def test_max_pooling_with_cupy():
    # 创建输入数据，使用 numpy 或 cupy
    xp = cp  # 切换到 cupy
    # 输入数据 (N, C, H, W)
    input_data = xp.random.rand(1, 1, 4, 4).astype(np.float32)  # 1x1x4x4 的随机数据
    print("Input data:\n", input_data)

    device = get_device_from_array(input_data)
    print(device)

    # 转换为 Tensor
    x = Tensor(input_data, requires_grad=True, device=device)

    # 定义一个最大池化层
    pool_h, pool_w, stride, padding = 2, 2, 2, 0
    max_pool = MaxPooling2D(pool_h, pool_w, stride, padding)

    # 执行前向传播
    output = max_pool.forward(x)
    print("Output data:\n", output.data)

    # 检查结果是否是 cupy 数据
    xp_result = get_array_module(output.data)
    assert xp_result is cp, "Test failed: Output is not on GPU (cupy)."
    print("Test passed: MaxPooling2D successfully runs on GPU using cupy.")

# 运行测试
test_max_pooling_with_cupy()
