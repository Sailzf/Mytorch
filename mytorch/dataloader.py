import math
from typing import Optional, Callable, List, Any, TypeVar
import numpy as np
from mytorch.dataset import Dataset
from mytorch.tensor import Tensor

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]

def default_collate(batch):
    """将一个批次的数据整理成批次张量"""
    if isinstance(batch[0], tuple):
        # 转置批次，从[(x1,y1), (x2,y2)...]变成([x1,x2...], [y1,y2...])
        transposed = zip(*batch)
        return [
            Tensor(np.stack([sample.data if isinstance(sample, Tensor) else sample 
                           for sample in samples]))
            for samples in transposed
        ]
    
    if isinstance(batch[0], Tensor):
        return Tensor(np.stack([x.data for x in batch]))
    
    return Tensor(np.array(batch))

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1,
                 shuffle: bool = True, collate_fn: Optional[_collate_fn_t] = None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_size = len(dataset)
        
        if collate_fn is None:
            collate_fn = default_collate
        
        self.collate_fn = collate_fn
        self.max_its = math.ceil(self.data_size / batch_size)
        self.it = 0
        self.indices = None
        self.reset()

    def reset(self):
        self.it = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.data_size)
        else:
            self.indices = np.arange(self.data_size)

    def __len__(self):
        return self.max_its

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.it >= self.max_its:
            self.reset()
            raise StopIteration

        start_idx = self.it * self.batch_size
        end_idx = min((self.it + 1) * self.batch_size, self.data_size)
        batch_indices = self.indices[start_idx:end_idx]
        
        # 收集批次数据
        batch = [self.dataset[i] for i in batch_indices]
        self.it += 1
        
        # 使用collate_fn整理批次数据
        return self.collate_fn(batch)

def prepare_mnist_data(root='data', backend='numpy', train=True, download=True):
    """通用的MNIST数据预处理函数
    
    Args:
        root (str): 数据存储的根目录
        backend (str): 后端类型，可选 'numpy' 或 'cupy'
        train (bool): 是否为训练集
        download (bool): 是否下载数据集
    
    Returns:
        dataset: 处理后的数据集
    """
    from torchvision import transforms, datasets
    import os
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 构建MNIST数据目录路径
    mnist_root = os.path.join(root, 'mnist')
    
    # 加载原始数据集
    mnist_dataset = datasets.MNIST(
        root=mnist_root,
        train=train,
        download=download,
        transform=transform
    )
    
    # 设置缓存目录和文件名
    cache_dir = os.path.join(mnist_root, f'processed_{backend}')
    cache_file = os.path.join(cache_dir, 'train' if train else 'test')
    
    # 根据后端类型选择相应的模块
    if backend == 'numpy':
        import numpy as np
        array_module = np
    elif backend == 'cupy':
        import cupy as cp
        array_module = cp
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    # 如果缓存文件存在，直接加载
    if os.path.exists(cache_file + '_data.npy') and os.path.exists(cache_file + '_targets.npy'):
        print(f"Loading cached dataset from {cache_file}")
        data = array_module.load(cache_file + '_data.npy')
        targets = array_module.load(cache_file + '_targets.npy')
    else:
        print("Converting dataset and creating cache...")
        data, targets = [], []
        for x, y in mnist_dataset:
            if backend == 'numpy':
                data.append(array_module.array(x))
            else:  # cupy
                data.append(array_module.array(x.numpy()))
            targets.append(y)
        
        data = array_module.stack(data)
        targets = array_module.array(targets)
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        # 保存为缓存文件
        array_module.save(cache_file + '_data.npy', data)
        array_module.save(cache_file + '_targets.npy', targets)
        print(f"Dataset cached to {cache_file}")
    
    from mytorch.tensor import Tensor
    from mytorch.dataset import MNISTDataset
    
    data = Tensor(data, requires_grad=False)
    targets = Tensor(targets, requires_grad=False)
    
    return MNISTDataset(data, targets)
