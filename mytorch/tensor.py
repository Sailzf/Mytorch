import contextlib
import heapq
import importlib
import inspect
import os
import time
from numbers import Number
from typing import Union, Tuple, List

import numpy as np

from mytorch import cuda
from mytorch.cuda import (
    Device,
    get_device_from_array,
    CpuDevice,
    GpuDevice,
    check_cuda_available,
    get_device,
    get_array_module,
    gpu_available
)

float_type = np.float32

# 设置显示精度
np.set_printoptions(precision=4)
# 抑制小数的科学计数法显示
np.set_printoptions(suppress=True)

NdArray = Union['np.ndarray', 'cuda.ndarray']

# 可以转换为数组的类型
Arrayable = Union[Number, NdArray, List]


def ensure_array(arrayable: Arrayable, dtype=None, device=None) -> NdArray:
    """
    :param arrayable:
    :param dtype:
    :return:
    """
    if device is not None:
        xp = device.xp
    else:
        xp = np

    if isinstance(arrayable, (Number, list)):
        # 让xp自己判断数据类型
        return xp.array(arrayable, dtype=dtype)
    elif isinstance(arrayable, (np.ndarray, cuda.ndarray)):
        if device is not None and get_array_module(arrayable) != xp:
            return device.transfer(arrayable)

    return arrayable


Tensorable = Union["Tensor", Number, NdArray]


def ensure_tensor(tensoralbe: Tensorable, device=None) -> "Tensor":
    '''
    确保是Tensor对象
    '''
    if isinstance(tensoralbe, Tensor):
        return tensoralbe

    return Tensor(tensoralbe, device=device)


class Config:
    debug = False
    backprop = True  # 是否需要计算并反向传播梯度


# 上下文管理器
# contextmanager 这个装饰器(decorator)接收一个生成器(generator)
# 该generator必须只yield一个值出来，该值会被用在with语句中，绑定到as后面的变量
# 我们这里只需要修改Config内部状态，不需要返回任何值，可以只加一个yield
@contextlib.contextmanager
def using_config(name, value):
    # 保存旧值
    old_value = getattr(Config, name)
    # 设置新值
    setattr(Config, name, value)
    try:
        yield
    finally:
        # 最终设回旧值
        setattr(Config, name, old_value)


def debug_mode():
    return using_config("debug", True)


def no_grad():
    return using_config("backprop", False)


class OpWrapper:
    '''
    支持反向传播的Debug
    '''

    def __init__(self, name, xs, backward=False):
        self.name = f"back_{name}" if backward else name
        self.xs = xs
        self.output = None
        threshold = int(os.getenv('THRESHOLD', 2))

        self.threshold = threshold

    def __enter__(self):
        if Config.debug:
            self.start = time.time()
        return self

    def __exit__(self, *junk):
        if Config.debug:
            end = (time.time() - self.start) * 1000
            if end > self.threshold:
                print(
                    f"{self.name:>20} : {end:>7.2f} ms {str([y.shape for y in self.xs]):>40} "
                    f"{'-> ' + str(self.output.shape) if self.output is not None else ''}"
                )


class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool = False, dtype=None, device: Device = None) -> None:
        '''
        初始化Tensor对象
        Args:
            data: 数据
            requires_grad: 是否需要计算梯度
            dtype: 数据类型，默认为None
            device: 设备类型 CpuDevice 或 GpuDevice
        '''
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            if device is None:
                device = data.device
            data = data.data

        if device is None:
            device = get_device_from_array(data)

        self._device = device
        self.creator = None
        self.generation = 0

        # data 是 NdArray
        self._data = ensure_array(data, dtype, self._device)

        self.requires_grad = requires_grad
        # 保存该Tensor的梯度
        self._grad = None  # 改成NdArray类型

        if self.requires_grad:
            self.zero_grad()

    #region 属性相关
    @property
    def grad(self):
        return self._grad

    @property
    def data(self) -> NdArray:
        return self._data

    @data.setter
    def data(self, new_data: NdArray) -> None:
        self._data = ensure_array(new_data, device=self.device)
        # 重新赋值后就没有梯度了
        self._grad = None

    # ****一些常用属性****
    @property
    def shape(self) -> Tuple:
        '''返回Tensor各维度大小的元素'''
        return self.data.shape

    @property
    def ndim(self) -> int:
        '''返回Tensor的维度个数'''
        return self.data.ndim

    @property
    def dtype(self):
        '''返回Tensor中数据的类型'''
        return self.data.dtype
    #endregion

    #region 数据类型相关
    def type(self, dtype):
        self.data = self.data.astype(dtype)
        return self

    def float(self):
        return self.type(np.float32)

    def short(self):
        return self.type(np.int16)

    def int(self):
        return self.type(np.int32)

    def long(self):
        return self.type(np.int64)

    def bool(self):
        return self.type(np.bool)
    #endregion

    #region 设备相关
    @property
    def device(self):
        return self._device

    @property
    def xp(self):
        '''返回当前tensor基于的numpy或cupy'''
        device = self.device
        return np if device is None else device.xp

    def to(self, device):
        device = get_device(device)
        # 如果设备一致了
        if get_device_from_array(self._data) == device:
            return self
        # 转移到设备上
        self._data = device.transfer(self.data)

        self._device = device

        if self._grad is not None:
            self._grad = Tensor(device.transfer(self._grad), device=device)

        return self

    def to_cpu(self):
        '''拷贝数据和梯度到CPU'''
        return self.to(CpuDevice())

    def to_gpu(self, device=None):
        '''拷贝数据和梯度到指定的GPU'''
        check_cuda_available()
        return self.to(get_device(device))
    #endregion

    #region 其他
    def zero_grad(self) -> None:
        '''
        将梯度初始化为0
        Returns:

        '''
        self._grad = self.xp.zeros_like(self.data)

    def __repr__(self) -> str:
        return f"Tensor(\n{self.data}, requires_grad={self.requires_grad}" \
               f"{', device:' + self.device.name if isinstance(self.device, GpuDevice) else ''})"

    def __len__(self) -> int:
        return len(self.data)

    def __gt__(self, other):
        other = ensure_tensor(other, self.device)
        return self.data > other.data

    def __lt__(self, other):
        other = ensure_tensor(other, self.device)
        return self.data < other.data

    def assign(self, x) -> "Tensor":
        '''将x的值赋予当前Tensor'''
        x = ensure_tensor(x, self.device)
        # 维度必须一致
        assert x.shape == self.shape
        self.data = x.data
        return self

    def size(self, dim=None) -> int:
        '''
        如果dim为None，返回Tensor中元素的个数 等同于np.prod(a.shape)；
        如果dim不为None，返回该维度上的元素个数
        Returns:
        '''

        return self.xp.size(self.data, dim)

    def array(self) -> NdArray:
        """转换为Numpy或Cupy数组"""
        return self.data

    def tolist(self):
        return self.data.tolist()

    def item(self) -> Number:
        '''将只有一个元素的Tensor转换为Python标量'''
        return self.array().item()

    #endregion

    # 计算图生成：每当 Tensor 对象是由某个操作生成的，都会调用 set_creator 方法，将生成它的操作登记为计算图的一部分。
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    def float_(self) -> "Tensor":
        self.data = self.data.astype(float_type)
        return self

    def is_floating_point(self):
        return isinstance(self.data, np.floating)

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> "Tensor":
        xp = self.device.xp
        self.data = xp.random.uniform(low, high, size=self.shape).astype(float_type)
        return self

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> "Tensor":
        xp = self.device.xp
        self.data = xp.random.normal(mean, std, size=self.shape).astype(float_type)
        return self

    def index_fill_(self, dim: int, index: "Tensor", value: float) -> "Tensor":
        xp = self.device.xp
        index = index.data

        axis = list(range(self.ndim))
        axis.remove(dim)
        # 扩展索引，put_along_axis需要索引数组和原数组维度一致
        index = xp.expand_dims(index, axis=axis)
        xp.put_along_axis(self.data, index, value, axis=dim)
        return self


    #region 作用是创建Tensor，
    #不需要先创建Tensor实例就可直接通过类调用，例如：x = Tensor.zeros(3, 4)  
    @classmethod
    def empty(cls, *shape, dtype=float_type, device=None, **kwargs):
        device = get_device(device)
        xp = device.xp
        return cls(xp.empty(*shape, dtype=dtype), device=device, **kwargs)
    
    @classmethod
    def emptyy(cls, *shape, dtype=float_type, device=None, **kwargs):
        device = get_device(device)
        xp = device.xp
        return cls(xp.empty(*shape, dtype=dtype), device=device, **kwargs)
    
    @classmethod
    def zeros(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.zeros(*shape, dtype=dtype), device=device, **kwargs)
    
    @classmethod
    def zeross(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = Device
        if (gpu_available):
            device = GpuDevice
        xp = device.xp
        return cls(xp.zeros(*shape, dtype=dtype), device=device, **kwargs)
    
    @classmethod
    def zeros_like(cls, t: "Tensor", **kwargs) -> "Tensor":
        return cls(t.xp.zeros(t.shape, dtype=t.dtype), device=t.device, **kwargs)

    @classmethod
    def ones(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.ones(shape=shape, dtype=dtype), device=device, **kwargs)

    @classmethod
    def ones_like(cls, t: "Tensor", **kwargs) -> "Tensor":
        return cls(t.xp.ones(shape=t.shape, dtype=t.dtype), device=t.device, **kwargs)

    @classmethod
    def randnn(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = Device
        if (gpu_available):
            device = GpuDevice
        xp = device.xp
        return cls(xp.random.randn(*shape).astype(dtype), device=device, **kwargs)
    
    @classmethod
    def randn(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.random.randn(*shape).astype(dtype), device=device, **kwargs)
    
    @classmethod
    def rand(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":  ##sailadd
        device = get_device(device)
        xp = device.xp
        return cls(xp.random.rand(*shape).astype(dtype), device=device, **kwargs)

    @classmethod
    def arange(cls, stop, start=0, step=1, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(data=xp.arange(stop=stop, start=start, step=step, dtype=dtype), device=device, **kwargs)

    @classmethod
    def eye(cls, dim, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.eye(dim).astype(dtype), device=device, **kwargs)

    @classmethod
    def full_like(cls, t: "Tensor", fill_value, dtype=float_type, requires_grad=False, device=None,
                  **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.full(t.shape, fill_value), device=device, **kwargs)

    @classmethod
    def uniform(cls, *shape, low: float = -1.0, high: float = 1.0,
                dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls((xp.random.uniform(low, high, size=shape)).astype(dtype), device=device, **kwargs)

    @classmethod
    def multinomial(cls, input: "Tensor", num_samples: int, replace=False) -> "Tensor":
        '''
        返回一个Tensor，每行包含num_samples个索引，从基于input对应行的多项式概率分布采样而来
        Args:
            input: 包含概率的输入，如果不是概率，那么会自动转换为概率
            num_samples: 生成的样本数
            replace: 是否为放回采样，默认为False
        '''

        size = input.size(-1)

        assert replace or num_samples <= size, "cannot sample n_sample > input.size(-1) samples without replacement"
        assert input.ndim <= 2, "prob_dist must be 1 or 2 dim"

        p = input.data / input.data.sum(-1, keepdims=True)
        xp = input.xp

        # 基于numpy.random.choice来实现multinomial
        if input.ndim == 1:
            return Tensor(xp.random.choice(xp.arange(size), replace=replace, size=num_samples, p=p),
                          device=input.device)
        else:
            # 如果input是2D，那么当成1D列表来处理
            ret = []
            for i in range(input.shape[0]):
                ret.append(xp.random.choice(xp.arange(size), replace=replace, size=num_samples, p=p[i]).tolist())

            return Tensor(ret, device=input.device)
    #endregion

    #region 实用操作
    def __getitem__(self, idxs) -> "Tensor":
        return self.slice(idxs)

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[key] = value
        return self

    def __ne__(self, other):
        return Tensor(self.data != ensure_tensor(other, self.device).data)

    @property
    def T(self) -> "Tensor":
        return self.transpose(axes=None)

    def _get_ops(self, name, *args, **kwargs):
        # 调用动态绑定的方法
        return self.__getattribute__(name, *args, **kwargs)

    def repeat(self, *sizes):
        if len(sizes) == 1:
            sizes = sizes[0]

        return self._get_ops('_repeat')(sizes)

    def reshape(self, *shape):
        if len(shape) == 1:
            shape = shape[0]

        return self._get_ops('_reshape')(shape)

    def view(self, *shape):
        return self.reshape(shape)

    def permute(self, *shape):
        return self._get_ops('transpose')(shape)

    def expand_dims(self, axis: int):
        return self._get_ops('expanddims')(axis)

    def __array__(self):
        return self.to_cpu().array()

    def detach(self):
        return Tensor(self)
    #endregion

    #region 反向传播,是自动求导的梯度累积过程；
    # 每当调用backward时，会根据链式法则，递归遍历其创建者（creator），将所有需要计算梯度的操作的梯度累加到grad属性中
    def backward(self, grad: NdArray = None, retain_grad=False, create_graph=False) -> None:
        '''
        实现Tensor的反向传播
        Args:
            grad: 如果不是标量，则需要传递梯度进来
            retain_grad: 是否保留梯度的中间变量
            create_graph: 整个计算梯度的过程是否也需要保留到计算图中，即double_backprop: todo 待实现

        Returns:

        '''
        # 只能在requires_grad=True的Tensor上调用此方法
        assert self.requires_grad, "called backward on tensor do not require grad"

        if not Config.backprop:
            return
        # 1. 初始化梯度,如果传递过来的grad为空
        if grad is None:
            if self.shape == ():
                # 设置梯度值为1，grad本身不需要计算梯度
                self._grad = self.xp.ones_like(self.data)

            else:
                # 如果当前Tensor得到不是标量，那么grad必须指定
                raise RuntimeError("grad must be specified for non scalar")
        else:
            self._grad = grad

        # 2. 初始化候选函数堆,将需要计算梯度的操作按照其生成顺序压入堆中，确保梯度计算顺序从后向前。
        funcs = []  # 候选函数堆
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                # heapq是小顶堆，为了实现大顶堆的效果，需要加一个负号
                heapq.heappush(funcs, (-f.generation, len(seen_set), f))
                seen_set.add(f)

        # 3. 将当前Tensor的创建者（self.creator）压入堆中
        add_func(self.creator)

        # 4. 从堆中弹出最小的元素，即最早生成的函数，进行梯度计算
        while funcs:
            _, _, f = heapq.heappop(funcs)
            # 获取输出对应的梯度，解决多个输出梯度不一致的问题
            gys = [output().grad for output in f.outputs]  # output 是 weakref

            with using_config('backprop', create_graph):
                with OpWrapper(f.__class__.__name__, gys, backward=True):
                    gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.requires_grad and gx is not None:
                        assert x.shape == gx.shape, f"grad shape must match tensor shape in {f!r}, {gx.shape!r} != {x.shape!r}"
                        if x.grad is None:
                            x._grad = gx
                        else:
                            x._grad = x._grad + gx  # 【根据链式法则的逻辑，累加多个路径传递的梯度】grad本身不需要计算梯度，所以普通NdArray即可

                        if x.creator is not None:
                            add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y()._grad = None

    def unchain_backward(self): # 反向传播时，将当前Tensor的创建者（self.creator）从候选函数堆中移除。
        if self.creator is not None:
            funcs = [self.creator]
            while funcs: # 通过递归地遍历计算图
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain() # 将x的创建者设为None，断开计算图中节点之间的依赖关系


def register(name, fxn):
    # 将运算类（如 Add, Mul）动态绑定到 Tensor 的魔法方法（如__add__）上
    def dispatch(*xs, **kwargs):
        # 分发函数，用于动态解析输入参数 xs 和 kwargs，并调用实际的操作函数 fxn。
        # fxn() 是一个实现特定操作的类（如 Add, Matmul 等）

        # device = [x for x in xs if isinstance(x, Tensor)][0].device
        # xs = [ensure_tensor(x, device) if not isinstance(x, Tensor) else x for x in xs]
        # xs = [x.data if isinstance(x, Tensor) else x for x in xs]
        return fxn()(*xs, **kwargs)

    if name in ["pow", "neg", "abs"]:
        setattr(Tensor, f"__{name}__", dispatch)

    if getattr(Tensor, name, None) is None:
        # 为Tensor添加属性，名为name，值为dispatch函数引用
        setattr(Tensor, name, dispatch)
    else:
        setattr(Tensor, f'_{name}', dispatch)

    # 这几个方法都有__xx__, __ixx__, __rxx__ 魔法方法
    if name in ["matmul"]:
        setattr(Tensor, f"__{name}__", dispatch)
        setattr(
            Tensor, f"__i{name}__", lambda self, x: self.assign(dispatch(self, x))
        )  # __i*__ 代表原地操作
        setattr(
            Tensor, f"__r{name}__", lambda self, x: dispatch(x, self)
        )  # __r*__ 代表 other在操作符前, self在操作符后


def _register_ops(namespace):
    for name, cls in inspect.getmembers(namespace, inspect.isclass):
        if name[0] != "_" and name != 'Tensor':
            # 注册所有Function的子类
            register(name.lower(), cls)


try:
    _register_ops(importlib.import_module("mytorch.ops"))
except ImportError as e:
    print(e)


# 注册运算类到Tensor的魔法方法上：
# 当执行这行代码时：from mytorch.tensor import Tensor
# Python 实际执行了：
# 1. 找到 mytorch 包
# 2. 执行 mytorch/__init__.py
# 3. 在 __init__.py 中:
#    - 导入 Tensor 类
#    - ops.install_ops() → 将运算类（如 Add, Mul）动态绑定到 Tensor 的方法（如 __add__、add）。
#    - 把运算的函数动态绑定到 Tensor 类上，一次性全部注册，等待+-*的调用
# 执行用户代码：如 a + b → 调用 Add.__call__ → 执行 Add.forward 完成前向计算。
# 反向传播：c.backward() → 调用 Add.backward → 计算输入梯度并存储到 a.grad, b.grad。

# 反向传播的实现：
# 1. 初始化梯度,如果传递过来的grad为空
# 2. 初始化候选函数堆,将需要计算梯度的操作按照其生成顺序压入堆中，确保梯度计算顺序从后向前。
# 3. 将当前Tensor的创建者（self.creator）压入堆中
# 4. 从堆中弹出最小的元素，即最早生成的函数，进行梯度计算
# 5. 反向传播时，将当前Tensor的创建者（self.creator）从候选函数堆中移除。

# # 假设有以下计算
# z = x + y     # generation = 1
# w = z * 2     # generation = 2
# v = w + 3     # generation = 3

# # 堆中的元素形式：(-generation, index, function)
# # 堆的内容：
# [(-3, 0, add_3),    # v = w + 3
#  (-2, 1, mul_2),    # w = z * 2
#  (-1, 2, add)]      # z = x + y

# # 出堆顺序：
# 1. (-3, 0, add_3)   # 先计算v的梯度
# 2. (-2, 1, mul_2)   # 再计算w的梯度
# 3. (-1, 2, add)     # 最后计算z的梯度