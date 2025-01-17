import numpy as np
import torch

import .functions as F
from .loss import NLLLoss
from .tensor import Tensor


def test_simple_nll_loss():
    x = np.array([[0, 1, 2, 3], [4, 0, 2, 1]], np.float32)
    t = np.array([3, 0]).astype(np.int32)

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(np.eye(x.shape[-1], dtype=np.int32)[t])  # 需要转换成one-hot向量

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    my_loss = NLLLoss()
    torch_loss = torch.nn.NLLLoss()

    # 先调用各自的log_softmax转换为对数概率
    ml = my_loss(F.log_softmax(mx), mt)
    tl = torch_loss(torch.log_softmax(tx, dim=-1, dtype=torch.float32), tt)
    assert np.allclose(ml.item(), tl.item())

    ml.backward()
    tl.backward()

    assert np.allclose(mx.grad, tx.grad)


def test_nll_loss():
    N, CLS_NUM = 100, 10  # 样本数，类别数
    x = np.random.randn(N, CLS_NUM)
    t = np.random.randint(0, CLS_NUM, (N,))

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(np.eye(x.shape[-1], dtype=np.int32)[t])  # 需要转换成one-hot向量

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    my_loss = NLLLoss()
    torch_loss = torch.nn.NLLLoss()

    # 先调用各自的log_softmax转换为对数概率
    ml = my_loss(F.log_softmax(mx), mt)
    tl = torch_loss(torch.log_softmax(tx, dim=-1, dtype=torch.float32), tt)

    assert np.allclose(ml.item(), tl.item())

    ml.backward()
    tl.backward()

    assert np.allclose(mx.grad, tx.grad)


def test_simple_nll_loss_class_indices():
    x = np.array([[0, 1, 2, 3], [4, 0, 2, 1]], np.float32)
    t = np.array([3, 0])

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(t)  # 类别索引

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    my_loss = NLLLoss()
    torch_loss = torch.nn.NLLLoss()
    # 先调用各自的log_softmax转换为对数概率
    ml = my_loss(F.log_softmax(mx), mt)
    tl = torch_loss(torch.log_softmax(tx, dim=-1, dtype=torch.float32), tt)
    assert np.allclose(ml.item(), tl.item())

    ml.backward()
    tl.backward()

    assert np.allclose(mx.grad, tx.grad)


def test_with_ignore_index():
    batch_size = 6
    num_classes = 4
    x = np.random.randn(batch_size, num_classes)
    t = np.random.randint(0, num_classes - 1, size=batch_size)  # 数值标签
    ignore_index = 2

    mx = Tensor(x, requires_grad=True)
    mt = Tensor(t)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.int64)

    my_loss = NLLLoss(ignore_index=ignore_index, reduction='mean')
    torch_loss = torch.nn.NLLLoss(ignore_index=ignore_index, reduction='mean')

    # 先调用各自的log_softmax转换为对数概率
    ml = my_loss(F.log_softmax(mx), mt)
    tl = torch_loss(torch.log_softmax(tx, dim=1, dtype=torch.float32), tt)
    assert np.allclose(ml.item(), tl.item())

    ml.backward()
    tl.backward()

    assert np.allclose(mx.grad, tx.grad)
