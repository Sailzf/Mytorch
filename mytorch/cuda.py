# import torch
import functools
from numbers import Number

import numpy

import mytorch


# 用于CPU的假的上下文管理器，该管理器啥也没做
class _FakeContext:
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


_fake_context = _FakeContext()


#   设备抽象和管理主要由 Device 抽象类及其子类实现：
#   - CpuDevice：
#     - 封装 NumPy 操作，xp 指向 numpy。
#     - 数据转换为 numpy.ndarray。
#   - GpuDevice：
#     - 封装 CuPy 操作，xp 指向 cupy。
#     - 数据转换为 cupy.ndarray，并处理 GPU 上的设备上下文。

# 具体使用中改变xp：
# 数据：用cupy创建的tensor，底层的相加等等操作会用cp
# 模型：model.to(device)后，model的实现会使用cp

class Device:
    '''
    通用的设备类
    1. 用来描述设备的行为，例如设备的名称 (name)、数组的转移 (transfer)、
    2. 设备上下文管理（支持 with 语法）。

    '''

    @property
    def xp(self):
        '''返回cupy或numpy'''
        raise NotImplementedError("Can not call from base class")

    @property
    def name(self):
        '''设备名称'''
        raise NotImplementedError("Can not call from base class")

    def transfer(self, array):
        '''将array转移到此设备'''
        raise NotImplementedError("Can not call from base class")

    def create_context(self):
        return _fake_context

    def __eq__(self, other):
        raise NotImplementedError("Can not call from base class")

    def __enter__(self):
        raise NotImplementedError("Can not call from base class")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError("Can not call from base class")


gpu_available = True

try:
    import cupy
    import cupyx
    from cupy import cuda
    from cupy import ndarray
    from cupy.cuda import Device as CudaDevice


except ImportError as e:
    print(e)
    # 当没有安装cupy时
    gpu_available = False


    # 创建一个假的ndarray
    class ndarray:
        @property
        def shape(self):
            pass

        @property
        def device(self):
            pass

        def get(self):
            pass

        def set(self, arr):
            pass


    # 假的CudaDevice
    class CudaDevice:
        def __init__(self, device=None):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


    # 假的cupy对象
    cupy = object()


class CpuDevice(Device):
    name = 'cpu'
    xp = numpy

    def __repr__(self):
        return "device(type='cpu')"

    def __eq__(self, other):
        return isinstance(other, CpuDevice)

    def transfer(self, array):
        if array is None:
            return None

        if isinstance(array, numpy.ndarray):
            return array

        if isinstance(array, (Number, list)):
            return numpy.asarray(array)

        if numpy.isscalar(array):
            return numpy.asarray(array)

        if isinstance(array, ndarray):
            return array.get()

        raise TypeError(f'Actual type{type(array)} cannot be converted to numpy.ndarray')


class GpuDevice(Device):
    xp = cupy

    def __init__(self, device: CudaDevice):
        check_cuda_available()

        assert isinstance(device, CudaDevice)
        super(GpuDevice, self).__init__()
        self.device = device

    @property
    def name(self):
        return f'cuda:{self.device.id}'

    @staticmethod
    def from_device_id(device_id: int = 0):
        check_cuda_available()

        return GpuDevice(cuda.Device(device_id))

    @staticmethod
    def from_array(array: ndarray):
        if isinstance(array, ndarray) and array.device is not None:
            return GpuDevice(array.device)
        return None

    def create_context(self):
        '''cuda.Device具有上下文管理器'''
        return cuda.Device(self.device.id)

    def transfer(self, array):
        if array is None:
            return None
        if isinstance(array, (Number, list)):
            # 将Number或number list转换为numpy数组
            array = numpy.asarray(array)

        if isinstance(array, ndarray):
            if array.device == self.device:
                return array
            is_numpy = False
        elif isinstance(array, numpy.ndarray):
            is_numpy = True
        else:
            raise TypeError(
                f'Actual type{type(array)} cannot be converted to cupy.ndarray'
            )
        if is_numpy:
            return cupy.asarray(array)
        # 拷贝到此设备
        return cupy.array(array, copy=True)

    def __eq__(self, other):
        return isinstance(other, GpuDevice) and other.device == self.device

    def __repr__(self):
        return f"device(type='cuda', index={self.device.id})"

# region 设备检查
    # is_available()：检查 GPU（cupy）是否可用。
    # check_cuda_available()：如果 cupy 不可用，抛出错误提示。
def is_available():
    return gpu_available


def check_cuda_available():
    if not gpu_available:
        raise RuntimeError('Install cupy first.')

# endregion 设备检查

# region 设备选择与获取
# get_device(device_desc)：
    # 根据描述符（如 'cpu'、'cuda'、'cuda:0'）选择设备。
    # 如果未指定设备描述符，默认为 CpuDevice。
# get_device_from_array(array)：
    # 根据数组所在设备返回 CpuDevice 或 GpuDevice。
# get_gpu_device_or_current(device)：
    # 获取指定的 GPU 设备或当前设备。

def get_device(device_desc) -> Device:
    '''
    根据device_desc描述符如 'cpu'、'cuda'、'cuda:0'选择设备(_Deivce)
    Args:
        device_desc: GpuDevice或CpuDevice
                     cpu -> CPU
                     cuda -> 默认显卡
                     cuda:1 -> 指定显卡1

    '''
    if device_desc is None:
        return CpuDevice()

    if isinstance(device_desc, Device):
        return device_desc

    if is_available() and isinstance(device_desc, CudaDevice):
        return GpuDevice(device_desc)

    if device_desc == 'cpu':
        return CpuDevice()

    if device_desc.startswith('cuda'):
        name, colon, device_id = device_desc.partition(':')
        if not colon:
            device_id = 0
        return GpuDevice.from_device_id(device_id)

    raise ValueError('Invalid argument.')

def get_device_from_array(array) -> Device:
    device = GpuDevice.from_array(array)
    if device is not None:
        return device

    return CpuDevice()

def get_gpu_device_or_current(device):  # -> Any | CudaDevice:
    check_cuda_available()

    if device is None:
        return cuda.Device()

    if isinstance(device, CudaDevice):
        return device

    if isinstance(device, int):
        return cuda.Device(device)

    raise ValueError('Invalid argument, only support `cuda.Device` or non-negative int')

def get_array_module(array):
    '''
    返回array对应的是numpy还是cupy
    '''
    if is_available():
        if isinstance(array, mytorch.Tensor):
            array = array.data
        return cupy.get_array_module(array)

    return numpy

# endregion 设备选择与获取

# using_device(device_desc)：上下文管理器，临时切换设备。
def using_device(device_desc):
    device = get_device(device_desc)
    return device.create_context() # 返回上下文管理器→_fake_context

    '''
    from cuda import using_device

    # 示例：在指定设备上执行操作
    with using_device("cuda:0"):
        # 在 GPU 设备 cuda:0 上执行的代码
        print("This code runs on GPU:0")

    # 示例：切换到 CPU 上执行
    with using_device("cpu"):
        # 在 CPU 上执行的代码
        print("This code runs on CPU")
    '''

# region 性能优化与 GPU 加速
    # memoize 装饰器，为elementwise提供缓存
        # GPU 可用时，使用 cupy.memoize 对函数进行缓存。
        # GPU 不可用时，提供假的装饰器。

def memoize(bool_for_each_device=False):
    if gpu_available:
        return cupy.memoize(bool_for_each_device)

    # 否则返回假的装饰器
    def dummy_decorator(f):
        @functools.wraps(f)
        def ret(*args, **kwargs):
            return f(*args, **kwargs)

        return ret

    return dummy_decorator

    # clear_memo()：清除 GPU 操作的缓存
def clear_memo():
    if gpu_available:
        cupy.clear_memo()


@memoize()
def elementwise(in_params, out_params, operation, name, **kwargs):
    '''
        调用cupy的ElementwiseKernel去加速GPU运行，注意需要编写C++代码，见 https://docs.cupy.dev/en/stable/user_guide/kernel.html

    Args:
        in_params: 输入参数
        out_params: 输出参数
        operation: 操作
        name: 名称
        **kwargs:

    Returns:

    '''

    check_cuda_available()
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, **kwargs)

    '''
    【例子：计算逐元素平方和】
    from cuda import elementwise

    # 定义一个在 GPU 上运行的逐元素操作内核
    square_sum_kernel = elementwise(
        'float32 x, float32 y',    # 输入参数
        'float32 z',               # 输出参数
        'z = x * x + y * y',       # CUDA 操作
        'square_sum'               # 内核名称
    )

    # 示例数据
    import cupy
    x = cupy.array([1, 2, 3], dtype='float32')
    y = cupy.array([4, 5, 6], dtype='float32')

    # 调用内核执行计算
    z = square_sum_kernel(x, y)
    print(z)  # 输出：[17. 29. 45.]，即 1^2+4^2, 2^2+5^2, 3^2+6^2


    '''

# endregion 性能优化与 GPU 加速