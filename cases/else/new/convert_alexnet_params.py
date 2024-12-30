import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import OrderedDict

from mytorch.tensor import Tensor
from mytorch.module import Module, Conv2D, Linear, ReLU, MaxPooling2D, Sequential, Dropout

class AlexNet(Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        
        self.features = Sequential(
            # conv1
            Conv2D(3, 64, kernel_size=(11, 11), stride=4, padding=2),
            ReLU(),
            MaxPooling2D(3, stride=2),
            # conv2
            Conv2D(64, 192, kernel_size=(5, 5), padding=2),
            ReLU(),
            MaxPooling2D(3, stride=2),
            # conv3
            Conv2D(192, 384, kernel_size=(3, 3), padding=1),
            ReLU(),
            # conv4
            Conv2D(384, 256, kernel_size=(3, 3), padding=1),
            ReLU(),
            # conv5
            Conv2D(256, 256, kernel_size=(3, 3), padding=1),
            ReLU(),
            MaxPooling2D(3, stride=2),
        )
        
        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(256 * 6 * 6, 4096),
            ReLU(),
            Dropout(p=0.5),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], 256 * 6 * 6)
        x = self.classifier(x)
        return x

def convert_conv_params(torch_conv, mytorch_conv):
    """转换卷积层参数"""
    # 权重转换
    weight_data = torch_conv.weight.detach().cpu().numpy()
    mytorch_conv.weight.data = weight_data
    
    # 偏置转换
    if torch_conv.bias is not None:
        bias_data = torch_conv.bias.detach().cpu().numpy()
        mytorch_conv.bias.data = bias_data

def convert_linear_params(torch_linear, mytorch_linear):
    """转换全连接层参数"""
    # 权重转换
    weight_data = torch_linear.weight.detach().cpu().numpy()
    mytorch_linear.weight.data = weight_data
    
    # 偏置转换
    if torch_linear.bias is not None:
        bias_data = torch_linear.bias.detach().cpu().numpy()
        mytorch_linear.bias.data = bias_data

def convert_sequential_params(torch_seq, mytorch_seq):
    """转换Sequential容器中的参数"""
    torch_modules = [m for m in torch_seq.modules() if not isinstance(m, nn.Sequential)]
    mytorch_modules = [m for m in mytorch_seq.modules() if not isinstance(m, Sequential)]
    
    # 跳过第一个元素（Sequential本身）
    torch_modules = torch_modules[1:]
    mytorch_modules = mytorch_modules[1:]
    
    for torch_module, mytorch_module in zip(torch_modules, mytorch_modules):
        if isinstance(torch_module, nn.Conv2d):
            convert_conv_params(torch_module, mytorch_module)
        elif isinstance(torch_module, nn.Linear):
            convert_linear_params(torch_module, mytorch_module)

def main():
    print("开始转换AlexNet参数...")
    
    # 1. 加载PyTorch预训练的AlexNet
    print("加载PyTorch预训练的AlexNet...")
    torch_alexnet = models.alexnet(pretrained=True)
    torch_alexnet.eval()
    
    # 2. 创建mytorch版本的AlexNet
    print("创建mytorch版本的AlexNet...")
    mytorch_alexnet = AlexNet()
    
    # 3. 转换特征提取层参数
    print("转换特征提取层参数...")
    convert_sequential_params(torch_alexnet.features, mytorch_alexnet.features)
    
    # 4. 转换分类器层参数
    print("转换分类器层参数...")
    convert_sequential_params(torch_alexnet.classifier, mytorch_alexnet.classifier)
    
    # 5. 保存转换后的参数
    print("保存转换后的参数...")
    mytorch_alexnet.save('alexnet_imagenet.pkl')
    
    print("参数转换完成！模型已保存到 alexnet_imagenet.pkl")

if __name__ == '__main__':
    main() 