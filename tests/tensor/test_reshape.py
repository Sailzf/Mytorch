import numpy as np

from .tensor import Tensor


def test_simple_reshape():
    x = Tensor(np.ones((3, 3)), requires_grad=True)
    z = x.reshape(-1)
    z.backward(np.ones(9))

    assert x.grad.tolist() == np.ones_like(x.data).tolist()


def test_reshape():
    x = Tensor.arange(9, requires_grad=True)
    z = x.reshape(3, 3)
    z.backward(np.ones((3, 3)))

    assert x.grad.tolist() == np.ones_like(x.data).tolist()


def test_matrix_reshape():
    x = Tensor(np.arange(12).reshape(2, 6).astype(np.float32), requires_grad=True)
    z = x.reshape(4, 3)

    z.backward(np.ones((4, 3)))

    assert x.grad.tolist() == np.ones_like(x.data).tolist()
