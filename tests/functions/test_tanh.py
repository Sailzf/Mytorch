import numpy as np
import torch

import .functions as F
from .tensor import Tensor, debug_mode


def test_simple_tanh():
    x = 2.0

    mx = Tensor(x, requires_grad=True, dtype=np.float32)
    y = F.tanh(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tanh(tx)

    assert np.allclose(y.data, ty.data)

    y.backward()
    ty.backward()

    assert np.allclose(mx.grad, tx.grad, rtol=1.e-4)


def test_tanh():
    x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)

    with debug_mode():
        mx = Tensor(x, requires_grad=True)
        y = F.tanh(mx)

        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tanh(tx)
        assert np.allclose(y.data, ty.data)

        y.sum().backward()
        ty.sum().backward()

        assert np.allclose(mx.grad, tx.grad, rtol=1.e-4)
