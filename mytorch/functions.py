# import torch
from typing import Tuple, Union, List

import numpy as np

from mytorch import cuda
from mytorch.cuda import get_array_module
from mytorch.ops import Function
from mytorch.tensor import Tensor, NdArray


# ----激活函数----
class ReLU(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        # y = xp.maximum(x, 0, dtype=x.dtype)
        y = xp.maximum(x, 0, dtype=float)
        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        if xp is np:
            return grad * (y > 0)
        else:
            return cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = y > 0 ? gy : (T)0', 'relu_bwd')(y, grad)


# def relu(x: Tensor) -> Tensor:
#     return x * (x > 0)
def relu(x: Tensor) -> Tensor:
    return ReLU()(x)


class LeakyRelu(Function):
    def forward(self, x: NdArray, slope: float = 0.01) -> NdArray:
        self.save_for_backward(x, slope)
        xp = get_array_module(x)
        return xp.maximum(x, 0) + slope * xp.minimum(x, 0)

    def backward(self, grad: NdArray) -> NdArray:
        x, slope = self.saved_tensors
        mask = np.array(x > 0).astype(grad.dtype)  # x > 0 : 1
        mask[mask <= 0] = slope  # x <=0 : slope
        return grad * mask


class ELU(Function):
    def forward(self, x: NdArray, alpha: float = 1) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x, alpha, xp)
        return xp.maximum(x, 0) + xp.minimum(0, alpha * (xp.exp(x) - 1))

    def backward(self, grad: NdArray) -> NdArray:
        x, alpha, xp = self.saved_tensors
        mask = xp.array(x > 0).astype(grad.dtype)  # x > 0 : 1 加上np.array 兼容标量
        indices = (mask <= 0)
        mask[indices] = alpha * xp.exp(x)[indices]  # x <= 0 :  alpha * exp(x)
        return grad * mask


class Sigmoid(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        # assert xp is np

        if xp is np:
            half = x.dtype.type(0.5)
            y = np.tanh(x * half) * half + half
        else:
            y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x * 0.5) * 0.5 + 0.5',
                'sigmoid_fwd')(x)

        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        if xp is np:
            one = y.dtype.type(1)
            return grad * y * (one - y)
        else:
            return cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * y * (1 - y)',
                'sigmoid_bwd')(y, grad)


def sigmoid(x: Tensor) -> Tensor:
    '''
        重写 return 1. / (1. + (-x).exp()) 加快速度

    Args:
        x:

    Returns:

    '''
    return Sigmoid()(x)


def logsigmoid(x: Tensor) -> Tensor:
    return sigmoid(x).log()


# def leaky_relu(x: Tensor, slope: float = 0.1) -> Tensor:
#    return x * (x > 0) + slope * x * (x < 0)
def leaky_relu(x: Tensor, slope: float = 0.01) -> Tensor:
    return LeakyRelu()(x, slope=slope)


# def elu(x: Tensor, a: float = 1) -> Tensor:
#    return x * (x > 0) + a * (x.exp() - 1) * (x < 0)
def elu(x: Tensor, alpha: float = 1) -> Tensor:
    return ELU()(x, alpha=alpha)


def swish(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def softplus(x: Tensor, beta: float = 1) -> Tensor:
    return (1 + (beta * x).exp()).log() / beta


class Tanh(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        if xp is np:
            y = np.tanh(x)
        else:
            y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x)',
                'tanh_fwd')(x)

        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        if xp is np:
            one = y.dtype.type(1)
            return grad * (one - y * y)
        else:
            return cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy *  (1 - y*y)',
                'tanh_bwd')(y, grad)


def tanh(x: Tensor) -> Tensor:
    return Tanh()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        y = x - x.max(axis=self.axis, keepdims=True)
        xp.exp(y, out=y)
        y /= y.sum(axis=self.axis, keepdims=True)

        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors

        gx = y * grad
        dx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * dx

        return gx


def softmax(x: Tensor, axis=-1):
    # b = x.max(axis=axis, keepdims=True)
    # y = (x - b).exp()
    # return y / y.sum(axis=axis, keepdims=True)

    return Softmax(axis=axis)(x)


def _logsumexp(x: NdArray, axis=-1):
    xp = get_array_module(x)
    b = x.max(axis=axis, keepdims=True)
    y = x - b
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    b += s
    return b


class LogSoftmax(Function):

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        y = x - _logsumexp(x, self.axis)
        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        return grad - xp.exp(y) * grad.sum(axis=self.axis, keepdims=True)


def log_softmax(x: Tensor, axis=-1):
    '''
    :param x: logits
    :param axis:
    :return:
    '''
    return LogSoftmax(axis=axis)(x)


def _reduction(errors: Tensor, method: str) -> Tensor:
    if method == "mean":
        loss = errors.sum() / errors.shape[0]
    elif method == "sum":
        loss = errors.sum()
    else:
        loss = errors

    return loss


def _softmax(x, axis=1):
    b = x.max(axis=axis, keepdims=True)
    y = (x - b).exp()
    return y / y.sum(axis=axis, keepdims=True)


class NLLLoss(Function):
    def __init__(self, ignore_index=-100, reduction: str = "mean"):
        """

        Args:
            ignore_index: 忽略的标签，可以是padding的标签(一般为0)，默认为-100
            reduction:
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target) -> NdArray:
        """
        Args:
            input: 对数概率 即 log_softmax 形状 (batch_size, num_classes)
            target:  类别索引 或 one-hot向量 形状为 (batch_size,) 或 (batch_size, num_classes)

        Returns:
        """
        xp = get_array_module(input)

        # 如果target是ont-hot向量，转换为一维向量
        if target.ndim > 1:
            target = xp.argmax(target, axis=1)

        batch_size, num_classes = input.shape
        # 根据ignore_index对标签进行忽略
        mask = (target != self.ignore_index).astype(int)

        errors = -xp.sum(input[xp.arange(batch_size), target] * mask, dtype=input.dtype)
        if self.reduction == 'mean':
            errors = xp.divide(errors, mask.sum(), dtype=input.dtype)

        self.save_for_backward(xp, target, input, batch_size, num_classes, mask)
        return errors

    def backward(self, grad: NdArray) -> NdArray:
        xp, target, input, batch_size, num_classes, mask = self.saved_tensors

        if target.ndim > 1:
            target = xp.argmax(target, axis=1)

        bigger_grad = xp.zeros((batch_size, num_classes), dtype=grad.dtype)
        bigger_grad[xp.arange(batch_size), target] = xp.divide(-mask, mask.sum(), dtype=input.dtype)

        return bigger_grad


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean", ignore_index=-100):
    # print("Input shape:", input.shape)  # 添加这行调试语句
    return NLLLoss(ignore_index, reduction)(input, target)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    '''

    :param input: logits
    :param target: 真实标签 0或1
    :param reduction:
    :return: binary cross entropy loss
    '''

    neg_abs = - abs(input)
    errors = input.clip(x_min=0) - input * target + (1 + neg_abs.exp()).log()

    return _reduction(errors, reduction)


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean", ignore_index=-100) -> Tensor:
    '''

    :param input: logits
    :param target: 真实标签one-hot向量 或 类别索引
    :param reduction:
    :return:
    '''
    # 先计算logsoftmax
    log_y = log_softmax(input)
    # 基于nll实现交叉熵损失
    return nll_loss(log_y, target, reduction, ignore_index)


class Dropout(Function):

    def __init__(self, p: float = 0.5):
        '''
        丢弃掩码 1代表保留，0代表丢弃 以1-p的概率生成输出为1伯努利分布，做了input的元素个数这么多次实验

        Args:
            p: dropout ratio 丢弃率
        '''
        super().__init__()
        self.p = p

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        if xp is np:
            scale = x.dtype.type(1. / 1 - self.p)
            flag = np.random.rand(*x.shape) >= self.p
            mask = scale * flag
            # 让输入乘上这个与之同shape的flag，然后除以1-p进行缩放，这样在测试时，可以原样输出
            y = x * mask
        else:
            rand = xp.random.rand(*x.shape, dtype=np.float32)
            scale = x.dtype.type(1. / (1 - self.p))
            mask, y = cuda.elementwise(
                'T x, R r, T scale, T p', 'T mask, T y',
                '''
                mask = (r >= p) * scale;
                y = x * mask;
                ''',
                'dropout_fwd',
            )(x, rand, scale, self.p)

        self.save_for_backward(mask)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        mask, = self.saved_tensors
        return grad * mask


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    x: 输入
    p: dropout ratio 丢弃率
    training: 是否为训练阶段
    """
    if training:
        return Dropout(p=p)(x)
    else:
        return x


class Embedding(Function):
    def forward(self, weight: NdArray, indices: NdArray) -> NdArray:
        self.save_for_backward(weight.shape, indices)
        return weight[indices]

    def backward(self, grad: NdArray) -> Tuple[NdArray, None]:
        w_shape, indices = self.saved_tensors

        xp = get_array_module(grad)

        bigger_grad = xp.zeros(w_shape, dtype=grad.dtype)

        if xp is np:
            np.add.at(bigger_grad, indices, grad)
        else:
            bigger_grad.scatter_add(indices, grad)

        # 因为它有两个输入，防止错误地拆开bigger_grad
        # indices 不需要梯度
        return bigger_grad, None


def embedding(weight: Tensor, indices: Tensor) -> Tensor:
    return Embedding()(weight, indices)


class MaskedSelect(Function):
    def forward(self, x: NdArray, mask: NdArray) -> NdArray:
        self.save_for_backward(x.shape, mask)
        return x[mask]

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, mask = self.saved_tensors
        xp = get_array_module(grad)

        bigger_grad = xp.zeros(x_shape, dtype=grad.dtype)

        bigger_grad[mask] = grad
        return bigger_grad


def masked_select(x: Tensor, mask):
    return MaskedSelect()(x, mask)


class Split(Function):
    '''Stack的逆操作'''

    def forward(self, inputs: NdArray, axis: int) -> NdArray:
        xp = get_array_module(inputs)
        xs = xp.split(inputs, inputs.shape[axis], axis)
        ys = [xp.squeeze(y, axis) for y in xs]  # 去掉维度axis
        self.save_for_backward(xp, axis, ys[0].shape, inputs.dtype)

        return tuple(ys)

    def backward(self, *grad: List[NdArray]) -> NdArray:
        xp, axis, shape, dtype = self.saved_tensors
        grads = [xp.zeros(shape, dtype) if g is None else g for g in grad]
        return xp.stack(grads, axis=axis)


def split(x: Tensor, axis: int = 0):
    return Split()(x, axis=axis)


unbind = split


class Stack(Function):
    '''
    在指定维度上进行堆叠，会增加维度
    维数：指有多少维
    维度：某个维的元素个数

    比如(2,3)的维数是2；第1个维度是2；第2个维度是3

    '''

    def forward(self, *inputs: Union[Tuple[NdArray, ...], List[NdArray]], axis: int) -> NdArray:
        xp = get_array_module(inputs[0])
        ret = xp.stack(inputs, axis=axis)
        self.save_for_backward(axis, xp)
        return ret

    def backward(self, grad: NdArray) -> NdArray:
        axis, xp = self.saved_tensors

        grads = xp.split(grad, grad.shape[axis], axis)
        grads = [xp.squeeze(g, axis) for g in grads]  # 去掉维度axis
        return tuple(grads)


def stack(xs: Union[Tuple[Tensor, ...], List[Tensor]], axis: int = 0):
    return Stack()(*xs, axis=axis)


class Cat(Function):
    '''
    在原有某一维度进行拼接，拼接的结果是Tensor的总维数不变，其中用于拼接的那一维度等于各分量维度之和
    '''

    def forward(self, *inputs: Union[Tuple[Tensor, ...], List[Tensor]], axis: int = -1) -> NdArray:
        xp = get_array_module(inputs[0])
        self.save_for_backward(inputs, axis, xp)
        return xp.concatenate(inputs, axis)

    def backward(self, grad: NdArray) -> NdArray:
        inputs, axis, xp = self.saved_tensors
        if len(inputs) == 1:
            return grad

        # 可能会不均分，所以大小可能不一致
        sizes = np.array(
            [x.shape[axis] for x in inputs[:-1]]
        ).cumsum()  # 计算累积和

        return tuple(xp.array_split(grad, sizes, axis))


def cat(xs: Union[Tuple[Tensor, ...], List[Tensor]], axis: int = 0):
    return Cat()(*xs, axis=axis)


class Chunk(Function):
    '''
    cat的逆操作，将Tensor沿某一维分开，chunks为分割的份数，axis为分割的维度
    '''

    def forward(self, inputs: NdArray, chunks: Union[int, NdArray], axis: int) -> Tuple[NdArray]:
        xp = get_array_module(inputs)
        ret = xp.array_split(inputs, chunks, axis)
        shapes = [x.shape for x in ret]
        self.save_for_backward(xp, axis, shapes, inputs.dtype)

        return tuple(ret)

    def backward(self, *grad: List[NdArray]) -> NdArray:
        xp, axis, shapes, dtype = self.saved_tensors
        grads = [xp.zeros(shape, dtype=dtype) if g is None else g for g, shape in zip(grad, shapes)]
        return xp.concatenate(grads, axis)


def chunk(input: Tensor, chunks: int, axis=0):
    return Chunk()(input, chunks=chunks, axis=axis)


class Flip(Function):
    def forward(self, inputs: NdArray, axis: Union[int, Tuple] = None) -> NdArray:
        xp = get_array_module(inputs)
        self.save_for_backward(axis, xp)
        return xp.flip(inputs, axis=axis)

    def backward(self, grad: NdArray) -> NdArray:
        axis, xp = self.saved_tensors
        return xp.flip(grad, axis=axis)


def flip(x: Tensor, axis: Union[int, Tuple] = None):
    return Flip()(x, axis=axis)


class Bmm(Function):
    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = self.saved_tensors
        return grad @ y.swapaxes(-2, -1), x.swapaxes(-2, -1) @ grad


def bmm(x: Tensor, y: Tensor):
    return Bmm()(x, y)


# 简单的norm实现
def norm(input: Tensor, p: int = 2, axis=None, keepdims=False):
    assert p in (1, 2), "Only support L2 normalization(p=2) and L1 normalization(p=1)"
    if p == 1:
        return abs(input).sum(axis=axis, keepdims=keepdims)
    else:
        return ((input ** 2).sum(axis=axis, keepdims=keepdims)) ** (1 / 2)


def cos_sim(u: Tensor, v: Tensor, axis=1):
    print(u.shape, v.shape)

    u_norm = norm(u, axis=axis)
    v_norm = norm(v, axis=axis)

    print(u_norm.shape, v_norm.shape)

    return u_norm @ v_norm.T
    #
    # fz = (u * v)
    # print(f'shape:{fz.shape}')
    # print(f'u_norm:{(u_norm * v_norm).shape}')
    # return (u / u_norm) * (v / v_norm)
