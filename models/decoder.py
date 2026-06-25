import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SP(nn.Module):
    def __init__(self, channel1, channel2, channel3, dilation_1=2, dilation_2=4, dilation_3=8):
        super(SP, self).__init__()

        self.conv1 = BasicConv2d(channel1, channel2, 1, padding=0)
        self.conv1_Dila = BasicConv2d(channel2, channel2, 3, padding=dilation_1, dilation=dilation_1)

        self.conv2 = BasicConv2d(channel1, channel2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel2, channel2, 3, padding=dilation_2, dilation=dilation_2)

        self.conv3 = BasicConv2d(channel1, channel2, 5, padding=2)
        self.conv3_Dila = BasicConv2d(channel2, channel2, 3, padding=dilation_3, dilation=dilation_3)

        self.conv_fuse = BasicConv2d(channel2 * 3, channel3, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)

        x2 = self.conv2(x)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv3(x)
        x3_dila = self.conv3_Dila(x3)

        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila, x3_dila), 1))
        return x_fuse

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class decoder(nn.Module):
    def __init__(self, dims=[64, 128, 320, 512], nclass=None):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.decoder4 = nn.Sequential(
            SP(dims[3], dims[3], dims[2]),
            TransBasicConv2d(dims[2], dims[2], kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(dims[2], nclass, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            SP(2 * dims[2], dims[2], dims[1]),
            TransBasicConv2d(dims[1], dims[1], kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(dims[1], nclass, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            SP(2 * dims[1], dims[1], dims[0]),
            TransBasicConv2d(dims[0], dims[0], kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(dims[0], nclass, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            SP(2 * dims[0], dims[0], 64),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2, padding=0, dilation=1, bias=False)
        )
        self.S1 = nn.Conv2d(64, nclass, 3, stride=1, padding=1)

    def forward(self, x4, x3, x2, x1):
        x4_up = self.decoder4(x4)
        s4 = self.S4(x4_up)

        x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        x1_up = self.upsample(x1_up)
        s1 = self.S1(x1_up)

        return s1, s2, s3, s4
