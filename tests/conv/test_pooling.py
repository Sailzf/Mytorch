import numpy as np
import cupy as cp
import .functions as F
from .tensor import Tensor, debug_mode, cuda
from .module import MaxPooling2D


# 定义一个简单的测试函数
def test_maxpooling2d():
    # 输入数据 (N, C, H, W)
    x_np = np.array([
        [[
            [1, 3, 2, 1],
            [4, 6, 5, 2],
            [7, 8, 9, 3],
            [1, 1, 2, 0]
        ]]
    ])  # 单通道 4x4 数据

    # 转为框架支持的 Tensor 类型
    x = Tensor(x_np)

    # 定义 MaxPooling2D 层
    pool = MaxPooling2D(pool_h=2, pool_w=2, stride=2)

    # 前向传播
    out = pool.forward(x)
    print("前向传播结果 (NumPy):")
    print(out.data)

    # 创建上游梯度 (N, C, out_h, out_w)
    dout_np = np.array([
        [[
            [1, 2],
            [3, 4]
        ]]
    ])
    dout = Tensor(dout_np)

    # # 反向传播
    # dx = pool.backward(dout)
    # print("\n反向传播梯度 (NumPy):")
    # print(dx.data)

    # 使用 CuPy 测试
    x_cp = cp.array(x_np)
    x_gpu = Tensor(x_cp)

    pool_gpu = MaxPooling2D(pool_h=2, pool_w=2, stride=2)
    out_gpu = pool_gpu.forward(x_gpu)

    print("\n前向传播结果 (CuPy):")
    print(out_gpu.data)

    # dout_cp = cp.array(dout_np)
    # dout_gpu = Tensor(dout_cp)

    # dx_gpu = pool_gpu.backward(dout_gpu)
    # print("\n反向传播梯度 (CuPy):")
    # print(cp.asnumpy(dx_gpu.data))  # 转回 NumPy 打印结果

# 运行测试
test_maxpooling2d()
