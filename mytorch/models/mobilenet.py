import mytorch.module as nn
import mytorch.functions as F
from mytorch.module import Module, Conv2D, BatchNorm2d
from mytorch.tensor import Tensor

class DepthwiseSeparableConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.pointwise = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        self.bn1 = BatchNorm2d(in_channels)
        self.bn2 = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class MobileNetBlock(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MobileNetBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)

class MobileNetV1(Module):
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super(MobileNetV1, self).__init__()
        
        def make_layers(in_channels, out_channels, stride):
            return MobileNetBlock(
                int(in_channels * width_multiplier),
                int(out_channels * width_multiplier),
                stride
            )

        self.first_conv = Conv2D(3, int(32 * width_multiplier), (3, 3), stride=2, padding=1)
        self.bn = BatchNorm2d(int(32 * width_multiplier))
        
        self.layers = nn.Sequential(
            make_layers(32, 64, 1),
            make_layers(64, 128, 2),
            make_layers(128, 128, 1),
            make_layers(128, 256, 2),
            make_layers(256, 256, 1),
            make_layers(256, 512, 2),
            *[make_layers(512, 512, 1) for _ in range(5)],
            make_layers(512, 1024, 2),
            make_layers(1024, 1024, 1)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(int(1024 * width_multiplier), num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 