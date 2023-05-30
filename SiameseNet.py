import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthWiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super(DepthWiseSeparableConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, bias=bias, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class SiameseNet(nn.Module):
    def __init__(self, architecture_config):
        super(SiameseNet, self).__init__()
        self.model = self._create_model(architecture_config)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    def _create_model(self, architecture_config):
        layer_list = list()
        for layer in architecture_config:
            layer_type, in_channels, out_channels, kernel_size, stride, padding = layer
            if layer_type == "BC":
                layer_list.append(
                    ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
                )
            elif layer_type == "DW":
                layer_list.append(
                    DepthWiseSeparableConvBlock(in_channels, out_channels, kernel_size, stride)
                )
        return nn.Sequential(*layer_list)

    def forward_one_input(self, x):
        x = self.model(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.forward_one_input(x1)
        x2 = self.forward_one_input(x2)
        return x1, x2

