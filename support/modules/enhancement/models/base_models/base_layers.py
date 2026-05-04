import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_fn(activation='relu'):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError("Unknown activation_fn")


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', is_batchnorm=False):
        super().__init__()
        if is_batchnorm:
            self.conv_relu = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                activation_fn(activation)
            )
        else:
            self.conv_relu = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                activation_fn(activation)
            )

    def forward(self, x):
        return self.conv_relu(x)


class ConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', is_batchnorm=False):
        super().__init__()
        if is_batchnorm:
            self.deconv_relu = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                activation_fn(activation)
            )
        else:
            self.deconv_relu = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                activation_fn(activation)
            )

    def forward(self, x):
        return self.deconv_relu(x)

class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.maxpool(x)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(x, dim=self.dim)


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            activation_fn(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv(x) + x


class MSIA(nn.Module):
    def __init__(self, in_channels, activation='relu'):
        super().__init__()
        self.conv1_1 = Conv2D(in_channels, in_channels, activation)
        self.conv1_2 = Conv2D(in_channels, in_channels, activation)
        self.conv2_1 = Conv2D(in_channels, in_channels, activation)
        self.conv2_2 = Conv2D(in_channels, in_channels, activation)
        self.conv3_1 = Conv2D(in_channels, in_channels, activation)
        self.conv3_2 = Conv2D(in_channels, in_channels, activation)
        self.conv_out = Conv2D(in_channels * 3, in_channels, activation)

    def forward(self, x, i_att):
        x1 = self.conv1_1(x)
        x1 = x1 * i_att
        x1 = self.conv1_2(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)

        x3 = self.conv3_1(x)
        x3 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=True)
        x3 = self.conv3_2(x3)
        x3 = F.interpolate(x3, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_out(x_out)
        return x_out + x
