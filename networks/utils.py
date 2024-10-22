import torch
import torch.nn as nn
import functools

############################# projector #############################
class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y_1 = self.conv(x)
        y_1 = self.bn(y_1)
        y_1 = self.relu(y_1)
        return y_1

class projectors(nn.Module):
    def __init__(self, input_nc=9, ndf=8, norm_layer=nn.BatchNorm2d):
        super(projectors, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = conv(input_nc, ndf)
        self.conv_2 = conv(ndf, ndf*2)
        self.final = nn.Conv2d(ndf*2, ndf*2, kernel_size=1)

    def forward(self, input):
        x_0 = self.conv_1(input)
        x_0 = self.pool(x_0)
        x_out = self.conv_2(x_0)
        x_out = self.pool(x_out)
        return x_out

