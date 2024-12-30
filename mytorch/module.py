import math
import operator
import pickle
from collections import OrderedDict
from itertools import chain, islice
from typing import List, Optional, Tuple, Dict, Iterable, Union, Iterator, Set

import mytorch.functions as F
from mytorch import init
from mytorch.paramater import Parameter
from mytorch.tensor import Tensor, no_grad, float_type
# from mytorch.rnn_utils import PackedSequence

import numpy as np
from mytorch.tensor import NdArray
from mytorch.cuda import get_array_module


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module:
    '''
    所有模型的基类
    '''

    training: bool

    def __init__(self) -> None:
        """
        调用super().__setattr__('a', a)而不是self.a=a防止调用Module.__setattr__的开销

        Module.__setattr__具有额外的对parameters,submodules的处理
        """
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        self._modules[name] = module

    def get_submodule(self, target: str) -> 'Module':
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms:
            mod = getattr(mod, item)

        return mod

    def get_parameter(self, target: str) -> Parameter:
        # 从最后一个.分隔
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        param: Parameter = getattr(mod, param_name)

        return param

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        '''

        Args:
            prefix:
            recurse: True，返回该module和所有submodule的参数；否则，仅返回该module的参数

        Yields:
            (string, Parameter): 包含名称和参数的元组

        '''
        gen = self._named_members(lambda module: module._parameters.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator['Module']:
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '',
                      remove_duplicate: bool = True):
        if memo is None:
            memo = set()

        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        return self

    def eval(self):
        """
        只会影响某些模型，比如Dropout和BatchNorm等
        :return:
        """
        return self.train(False)

    def state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".")

        return destination

    def _load_from_state_dict(self, state_dict, prefix):
        local_name_params = self._parameters.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                with no_grad():
                    # 赋值给param
                    param.data = input_param

    def load_state_dict(self, state_dict):
        state_dict = OrderedDict(state_dict)

        def load(module, local_state_dict, prefix=""):
            module._load_from_state_dict(local_state_dict, prefix)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load

    def save(self, path='model.pkl'):
        state_dict = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
            print(f"Save module to {path}")

    def load(self, path='model.pkl'):
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for key, param in self._parameters.items():
            if param is None:
                continue

            with no_grad():
                param_applied = fn(param)

            out_param = Parameter(param_applied)
            self._parameters[key] = out_param

        return self

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)

        fn(self)
        return self

    def to_gpu(self, device):
        return self._apply(lambda t: t.to_gpu(device))

    def to_cpu(self):
        return self._apply(lambda t: t.to_cpu())

    def to(self, device):
        return self._apply(lambda t: t.to(device))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        '''
        通过该魔法方法注册属性到Module中
        Args:
            name:
            value:

        Returns:

        '''

        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{value}' as parameter '{name}' "
                                "(torch.nn.Parameter or None expected)")
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign '{value}' as child module '{name}' "
                                    "(torch.nn.Module or None expected)")
                modules[name] = value
            else:
                super().__setattr__(name, value)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate(self):
        state = self.__dict__.copy()
        return state

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def freeze_parameters(self, names: Optional[List[str]] = None) -> None:
        """
        冻结指定参数或所有参数

        Args:
            names: 要冻结的参数名列表，如果为None则冻结所有参数
        """
        if names is None:
            # 冻结所有参数
            for param in self.parameters():
                param.requires_grad = False
        else:
            # 冻结指定参数
            for name in names:
                param = self.get_parameter(name)
                param.requires_grad = False

    def unfreeze_parameters(self, names: Optional[List[str]] = None) -> None:
        """
        解冻指定参数或所有参数

        Args:
            names: 要解冻的参数名列表，如果为None则解冻所有参数
        """
        if names is None:
            # 解冻所有参数
            for param in self.parameters():
                param.requires_grad = True
        else:
            # 解冻指定参数
            for name in names:
                param = self.get_parameter(name)
                param.requires_grad = True

    def is_frozen(self) -> bool:
        """
        检查模型是否被冻结（所有参数的requires_grad都为False）
        
        Returns:
            bool: 如果所有参数都被冻结返回True，否则返回False
        """
        return all(not param.requires_grad for param in self.parameters())


class Linear(Module):
    r"""
         对给定的输入进行线性变换: :math:`y=xA^T + b`

        Args:
            in_features: 每个输入样本的大小
            out_features: 每个输出样本的大小
            bias: 是否含有偏置，默认 ``True``
            device: CpuDevice或GpuDevice
            dtype: np.dtype
        Shape:
            - Input: `(*, H_in)` 其中 `*` 表示任意维度，包括none,这里 `H_{in} = in_features`
            - Output: :math:`(*, H_out)` 除了最后一个维度外，所有维度的形状都与输入相同，这里H_out = out_features`
        Attributes:
            weight: 可学习的权重，形状为 `(out_features, in_features)`.
            bias:   可学习的偏置，形状 `(out_features)`.
        """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.empty((out_features, in_features)), **factory_kwargs)
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features), **factory_kwargs)  # sailadd
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.weight)  # 默认采用kaiming初始化

    def forward(self, input: Tensor) -> Tensor:
        x = input @ self.weight.T
        if self.bias is not None:
            x = x + self.bias

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Sequential(Module):
    """
    顺序容器。按顺序包含多个模块并按顺序执行。
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        size = len(self)
        idx = operator.index(idx)
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input) -> Tensor:
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> 'Sequential':
        self.add_module(str(len(self)), module)
        return self


class ModuleList(Module):
    _modules: Dict[str, Module]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super(ModuleList, self).__init__()

        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        '''返回绝对值索引'''
        idx = operator.index(idx)  # 转换为int
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: int) -> Union[Module, 'ModuleList']:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))

        # 为了保留编号，在删除 self._modules 之后使用modules重新构建它
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> 'ModuleList':
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> 'ModuleList':
        combined = ModuleList()
        # chain将self, other变成一个可迭代对象
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    def insert(self, index: int, module: Module) -> None:
        '''
        在给定index之前插入module
        Args:
            index: 要插入的索引
            module: 要插入的module
        '''
        # 数组的插入算法，我们需要维护str(i)
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - i)]

        self._modules[str(index)] = module

    def append(self, module: Module) -> 'ModuleList':
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> 'ModuleList':
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


# ****激活函数作为Module实现****
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


# Dropout
class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        """
        :param p: 丢弃率
        """

        super(Dropout, self).__init__()

        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training)

    def extra_repr(self) -> str:
        return f'p={self.p}'


# CNN
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 使用 uniform 替代 rand
        self.weight = Parameter(
            Tensor.uniform(out_channels, in_channels, kernel_size, kernel_size, low=-0.1, high=0.1)
        )
        if bias:
            self.bias = Parameter(Tensor.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Apply padding if necessary
        if self.padding > 0:
            x_padded = Tensor.zeros((batch_size, self.in_channels, 
                                     height + 2 * self.padding, width + 2 * self.padding))
            x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = x_padded

        # 初始化输出
        out = Tensor.zeros((batch_size, self.out_channels, out_height, out_width))
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        start_h = i * self.stride
                        start_w = j * self.stride
                        end_h = start_h + self.kernel_size
                        end_w = start_w + self.kernel_size
                        region = x[b, :, start_h:end_h, start_w:end_w]
                        out[b, oc, i, j] = (region * self.weight[oc]).sum()
                        if self.bias is not None:
                            out[b, oc, i, j] += self.bias[oc]
        return out

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Apply padding if necessary
        if self.padding > 0:
            x_padded = Tensor.zeros((batch_size, channels, 
                                     height + 2 * self.padding, width + 2 * self.padding))
            x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = x_padded

        # 初始化输出
        out = Tensor.zeros((batch_size, channels, out_height, out_width))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        start_h = i * self.stride
                        start_w = j * self.stride
                        end_h = start_h + self.kernel_size
                        end_w = start_w + self.kernel_size
                        region = x[b, c, start_h:end_h, start_w:end_w]
                        out[b, c, i, j] = region.max()
        return out






class Conv2D(Module):
    """
    二维卷积层，支持前向传播，支持 numpy 和 cupy，支持分组卷积。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: int = 1, padding: int = 0, groups: int = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        """
        初始化卷积层。
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数（滤波器个数）
            kernel_size (Tuple[int, int]): 卷积核的大小 (高度, 宽度)
            stride (int, optional): 步长，默认为1
            padding (int, optional): 填充，默认为0
            groups (int, optional): 分组数，默认为1
        """
        super().__init__()
        assert in_channels % groups == 0, "输入通道数必须能被分组数整除"
        assert out_channels % groups == 0, "输出通道数必须能被分组数整除"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # 初始化权重和偏置
        kh, kw = kernel_size
        scale = (2 / (in_channels * kh * kw // groups)) ** 0.5  # Kaiming初始化
        self.weight = Parameter(Tensor.randn(out_channels, in_channels // groups, kh, kw) * scale, **factory_kwargs)
        self.bias = Parameter(Tensor.zeros(out_channels, requires_grad=True), **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入数据，形状为 (N, C, H, W)
            
        Returns:
            Tensor: 输出数据，形状为 (N, out_channels, out_h, out_w)
        """
        xp = get_array_module(x.data)
        N, C, H, W = x.shape
        FN, FC, FH, FW = self.weight.shape
        
        # 计算输出的高度和宽度
        out_h = (H + 2 * self.padding - FH) // self.stride + 1
        out_w = (W + 2 * self.padding - FW) // self.stride + 1

        if self.groups == 1:
            # 标准卷积
            col = im2col(x.data, FH, FW, self.stride, self.padding, xp)
            col_W = self.weight.data.reshape(FN, -1).T
            out = xp.dot(col, col_W) + self.bias.data
        else:
            # 分组卷积
            out = xp.zeros((N, self.out_channels, out_h, out_w))
            for g in range(self.groups):
                # 选择当前组的输入和权重
                x_g = x[:, g*(C//self.groups):(g+1)*(C//self.groups)]
                w_g = self.weight[g*(FN//self.groups):(g+1)*(FN//self.groups)]
                b_g = self.bias[g*(FN//self.groups):(g+1)*(FN//self.groups)]
                
                # 对当前组进行卷积
                col = im2col(x_g.data, FH, FW, self.stride, self.padding, xp)
                col_W = w_g.data.reshape(-1, FH*FW*FC).T
                out_g = xp.dot(col, col_W) + b_g.data.reshape(1, -1)
                out_g = out_g.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
                
                # 将结果放入对应位置
                out[:, g*(FN//self.groups):(g+1)*(FN//self.groups)] = out_g

        out = out.reshape(N, self.out_channels, out_h, out_w)
        return Tensor(out, requires_grad=True)

    
class MaxPooling2D(Module):
    """
    最大池化层，支持前向传播，反向传播由计算图自动完成。
    """

    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, padding: int = 0) -> None:
        """
        初始化最大池化层。
        
        Args:
            pool_h (int): 池化窗口的高度
            pool_w (int): 池化窗口的宽度
            stride (int): 步长
            padding (int): 填充
        """
        super().__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x (Tensor): 输入数据，形状为 (N, C, H, W)

        Returns:
            Tensor: 输出数据，形状为 (N, C, out_h, out_w)
        """
        xp = get_array_module(x.data)
        N, C, H, W = x.shape

        # 计算输出的高度和宽度
        out_h = (H + 2 * self.padding - self.pool_h) // self.stride + 1
        out_w = (W + 2 * self.padding - self.pool_w) // self.stride + 1

        # 将输入展开为二维矩阵
        col = im2col(x.data, self.pool_h, self.pool_w, self.stride, self.padding, xp)

        # 调整形状以适应池化窗口
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 找到最大值
        out = xp.max(col, axis=1)

        # 调整形状为 (N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        # 返回 Tensor 对象
        return Tensor(out, requires_grad=x.requires_grad)


def im2col(input_data: NdArray, filter_h: int, filter_w: int, stride: int, pad: int, xp) -> NdArray:
    """
    将输入数据展开为二维矩阵，支持 numpy 和 cupy。
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 使用 xp 确保一致性
    img = xp.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col: NdArray, input_shape: Tuple[int, int, int, int], filter_h: int, filter_w: int,
           stride: int, pad: int, xp) -> NdArray:
    """
    将二维矩阵还原为原始输入形状，支持 numpy 和 cupy。
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = xp.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class BatchNorm2d(Module):
    """
    Batch Normalization for 2D inputs (NCHW)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可训练参数
        self.weight = Parameter(Tensor.ones(num_features), **factory_kwargs)  # gamma
        self.bias = Parameter(Tensor.zeros(num_features), **factory_kwargs)   # beta
        
        # 运行时统计量，不需要训练
        self.register_buffer('running_mean', Tensor.zeros(num_features))
        self.register_buffer('running_var', Tensor.ones(num_features))
        
        # 初始化统计计数
        self.num_batches_tracked = 0

    def forward(self, x: Tensor) -> Tensor:
        xp = get_array_module(x.data)
        N = x.shape[0] * x.shape[2] * x.shape[3]  # 样本数 * 高度 * 宽度
        
        if self.training:
            # 计算当前batch的均值和方差
            batch_mean = x.mean(axis=(0, 2, 3))
            # 使用无偏估计计算方差
            batch_var = x.var(axis=(0, 2, 3)) * N / (N - 1)
            
            # 更新运行时统计量
            if self.num_batches_tracked == 0:
                self.running_mean = batch_mean
                self.running_var = batch_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            self.num_batches_tracked += 1
            
            # 使用当前batch的统计量进行归一化
            mean = batch_mean
            var = batch_var
        else:
            # 评估模式使用运行时统计量
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_normalized = (x - mean.reshape(1, -1, 1, 1)) / xp.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        
        # 缩放和平移
        return self.weight.reshape(1, -1, 1, 1) * x_normalized + self.bias.reshape(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return (f'{self.num_features}, '
                f'eps={self.eps}, '
                f'momentum={self.momentum}')

class AdaptiveAvgPool2D(Module):
    """
    自适应平均池化层
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        out_h, out_w = self.output_size
        
        # 计算池化窗口大小和步长
        stride_h = height // out_h
        stride_w = width // out_w
        kernel_h = height - (out_h - 1) * stride_h
        kernel_w = width - (out_w - 1) * stride_w
        
        # 使用im2col进行池化操作
        xp = get_array_module(x.data)
        col = im2col(x.data, kernel_h, kernel_w, stride_h, 0, xp)
        col = col.reshape(-1, kernel_h * kernel_w)
        out = xp.mean(col, axis=1)
        out = out.reshape(batch_size, out_h, out_w, channels).transpose(0, 3, 1, 2)
        
        return Tensor(out, requires_grad=x.requires_grad)
