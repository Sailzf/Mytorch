import torch

# 检查 CUDA Toolkit 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Toolkit Available: {cuda_available}")

# 检查 cuDNN 版本
if cuda_available:
    cudnn_version = torch.backends.cudnn.version()
    print(f"cuDNN Version: {cudnn_version}")
else:
    print("cuDNN not detected, CUDA is unavailable.")