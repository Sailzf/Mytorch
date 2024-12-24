from mytorch.tensor import Tensor # 必须在第一行，先执行_register_ops
import mytorch.functions
import mytorch.init
import mytorch.ops
from mytorch import cuda

ops.install_ops()

from mytorch.tensor import no_grad
from mytorch.tensor import ensure_tensor
from mytorch.tensor import ensure_array
from mytorch.tensor import float_type
from mytorch.tensor import debug_mode

from mytorch import module as nn
from mytorch import optim