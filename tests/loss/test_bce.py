import numpy as np
import torch

from .loss import BCELoss
from .tensor import Tensor


def test_bce():
    x = np.random.randn(3)
    t = np.random.random(3)
    mx = Tensor(x, requires_grad=True)
    mt = Tensor(t)

    tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    tt = torch.tensor(t, dtype=torch.float32)

    my_loss = BCELoss()
    torch_loss = torch.nn.BCEWithLogitsLoss()

    ml = my_loss(mx, mt)
    tl = torch_loss(tx, tt)

    assert np.allclose(ml.item(), tl.item())

    ml.backward()
    tl.backward()

    assert np.allclose(mx.grad, tx.grad)
