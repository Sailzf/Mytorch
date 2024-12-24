from mytorch.tensor import Tensor
import numpy as np


class Dataset:

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), \
            "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return tuple(tensor[index] for tensor in self.tensors)
        # 支持numpy数组索引
        return tuple(tensor.array()[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


class MNISTDataset(Dataset):
    """专门用于MNIST数据集的包装器"""
    def __init__(self, data, targets):
        self.data = data  # shape: (N, 1, 28, 28)
        self.targets = targets  # shape: (N,)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        # 确保返回正确的形状
        if isinstance(index, int):
            x = x.reshape(1, 28, 28)
        return x, y

    def __len__(self):
        return len(self.data)
