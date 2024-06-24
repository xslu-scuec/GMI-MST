# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class ConvBlock(nn.Module):
    def __init__(self, dim, in_channel, out_channel, kernel_size=3, stride=1) -> None:
        super(ConvBlock, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        self.input_conv = conv_fn(in_channel, out_channel, kernel_size, stride, padding=1)
        self.output_conv = conv_fn(out_channel, out_channel, kernel_size, stride, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.input_conv(x)
        out = self.activation(out)
        out = self.output_conv(out)
        out = self.activation(out)
        return out

class ConvBlockZ2d(nn.Module):
    def __init__(self, dim, in_channel, out_channel, kernel_size=(3,3,1), stride=1) -> None:
        super(ConvBlockZ2d, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        self.input_conv = conv_fn(in_channel, out_channel, kernel_size, stride, padding=(1,1,0))
        self.output_conv = conv_fn(out_channel, out_channel, kernel_size, stride, padding=(1,1,0))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.input_conv(x)
        out = self.activation(out)
        out = self.output_conv(out)
        out = self.activation(out)
        return out


class NestedUnet(nn.Module):
    def __init__(self, in_shape, in_channel, out_channel, nb_features=None, deep_supervision=True) -> None:
        super(NestedUnet, self).__init__()

        self.deep_supervision = deep_supervision

        ndim = len(in_shape)
        assert ndim in [2, 3], 'the dim of the in_shape should be one of 2, or 3. found: %d' % ndim

        conv_fn = getattr(nn, "Conv{0}d".format(ndim))
        pool_fn = getattr(nn, "MaxPool{0}d".format(ndim))
        upsample_fn = getattr(nn, "ConvTranspose{0}d".format(ndim))

        self.maxpool = pool_fn(2)

        if nb_features is None:
            nb_features = [16, 32, 64, 128, 256]   # 512 is bottom neck layer

        self.conv0_0 = ConvBlock(ndim, in_channel, nb_features[0])
        self.conv1_0 = ConvBlock(ndim, nb_features[0], nb_features[1])
        self.conv2_0 = ConvBlock(ndim, nb_features[1], nb_features[2])
        self.conv3_0 = ConvBlock(ndim, nb_features[2], nb_features[3])
        self.conv4_0 = ConvBlock(ndim, nb_features[3], nb_features[4])

        self.conv0_1 = ConvBlock(ndim, nb_features[0] + nb_features[0], nb_features[0])
        self.conv1_1 = ConvBlock(ndim, nb_features[1] + nb_features[1], nb_features[1])
        self.conv2_1 = ConvBlock(ndim, nb_features[2] + nb_features[2], nb_features[2])
        self.conv3_1 = ConvBlock(ndim, nb_features[3] + nb_features[3], nb_features[3])

        self.conv0_2 = ConvBlock(ndim, nb_features[0] * 2 + nb_features[0], nb_features[0])
        self.conv1_2 = ConvBlock(ndim, nb_features[1] * 2 + nb_features[1], nb_features[1])
        self.conv2_2 = ConvBlock(ndim, nb_features[2] * 2 + nb_features[2], nb_features[2])

        self.conv0_3 = ConvBlock(ndim, nb_features[0] * 3 + nb_features[0], nb_features[0])
        self.conv1_3 = ConvBlock(ndim, nb_features[1] * 3 + nb_features[1], nb_features[1])

        self.conv0_4 = ConvBlock(ndim, nb_features[0] * 4 + nb_features[0], nb_features[0])

        self.up0_1 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_1 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_1 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)
        self.up3_1 = upsample_fn(nb_features[4], nb_features[3], kernel_size=2, stride=2)

        self.up0_2 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_2 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_2 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)

        self.up0_3 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_3 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)

        self.up0_4 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)

        if deep_supervision:
            self.final_1 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_2 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_3 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_4 = conv_fn(nb_features[0], out_channel, kernel_size=1)
        else:
            self.final = conv_fn(nb_features[0], out_channel, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up0_1(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up0_2(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1_2(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up0_3(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1_3(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up0_4(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final_1(x0_1)
            output2 = self.final_2(x0_2)
            output3 = self.final_3(x0_3)
            output4 = self.final_4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return [x0_0, x0_1, output]


class ENestedUnet(nn.Module):
    def __init__(self, in_shape, in_channel, out_channel, nb_features=None, deep_supervision=True) -> None:
        super(ENestedUnet, self).__init__()

        self.deep_supervision = deep_supervision

        ndim = len(in_shape)
        assert ndim in [2, 3], 'the dim of the in_shape should be one of 2, or 3. found: %d' % ndim

        conv_fn = getattr(nn, "Conv{0}d".format(ndim))
        pool_fn = getattr(nn, "MaxPool{0}d".format(ndim))
        upsample_fn = getattr(nn, "ConvTranspose{0}d".format(ndim))

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = pool_fn(2)

        if nb_features is None:
            nb_features = [16, 32, 64, 128, 256]   # 512 is bottom neck layer

        self.conv0_0_0 = conv_fn(in_channel, nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_0_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_0_0 = conv_fn(nb_features[0], nb_features[1], kernel_size=3, stride=1, padding=1)
        self.conv1_0_1 = conv_fn(nb_features[1], nb_features[1], kernel_size=3, stride=1, padding=1)
        self.conv2_0 = ConvBlock(ndim, nb_features[1], nb_features[2])
        self.conv3_0 = ConvBlock(ndim, nb_features[2], nb_features[3])
        self.conv4_0 = ConvBlock(ndim, nb_features[3], nb_features[4])

        self.conv0_1 = ConvBlock(ndim, nb_features[0] + nb_features[0], nb_features[0])
        self.conv1_1 = ConvBlock(ndim, nb_features[1] + nb_features[1], nb_features[1])
        self.conv2_1 = ConvBlock(ndim, nb_features[2] + nb_features[2], nb_features[2])
        self.conv3_1 = ConvBlock(ndim, nb_features[3] + nb_features[3], nb_features[3])

        self.conv0_2 = ConvBlock(ndim, nb_features[0] * 2 + nb_features[0], nb_features[0])
        self.conv1_2 = ConvBlock(ndim, nb_features[1] * 2 + nb_features[1], nb_features[1])
        self.conv2_2 = ConvBlock(ndim, nb_features[2] * 2 + nb_features[2], nb_features[2])

        self.conv0_3 = ConvBlock(ndim, nb_features[0] * 3 + nb_features[0], nb_features[0])
        self.conv1_3 = ConvBlock(ndim, nb_features[1] * 3 + nb_features[1], nb_features[1])

        self.conv0_4 = ConvBlock(ndim, nb_features[0] * 4 + nb_features[0], nb_features[0])

        self.up0_1 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_1 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_1 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)
        self.up3_1 = upsample_fn(nb_features[4], nb_features[3], kernel_size=2, stride=2)

        self.up0_2 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_2 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_2 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)

        self.up0_3 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_3 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)

        self.up0_4 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)

        if deep_supervision:
            self.final_1 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_2 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_3 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_4 = conv_fn(nb_features[0], out_channel, kernel_size=1)
        else:
            self.final = conv_fn(nb_features[0], out_channel, kernel_size=1)

    def forward(self, x):
        x0_0_0 = self.conv0_0_0(x)
        x0_0_1 = self.activation(x0_0_0)
        x0_0_2 = self.conv0_0_1(x0_0_1)
        x0_0_3 = self.activation(x0_0_2)
        x1_0_0 = self.conv1_0_0(self.maxpool(x0_0_3))
        x1_0_1 = self.activation(x1_0_0)
        x1_0_2 = self.conv1_0_1(x1_0_1)
        x1_0_3 = self.activation(x1_0_2)
        x0_1 = self.conv0_1(torch.cat([x0_0_3, self.up0_1(x1_0_3)], dim=1))

        x2_0 = self.conv2_0(self.maxpool(x1_0_3))
        x1_1 = self.conv1_1(torch.cat([x1_0_3, self.up1_1(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0_3, x0_1, self.up0_2(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0_3, x1_1, self.up1_2(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0_3, x0_1, x0_2, self.up0_3(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0_3, x1_1, x1_2, self.up1_3(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0_3, x0_1, x0_2, x0_3, self.up0_4(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final_1(x0_1)
            output2 = self.final_2(x0_2)
            output3 = self.final_3(x0_3)
            output4 = self.final_4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return [x0_0_0, x1_0_0, output]


class FNestedUnet(nn.Module):
    def __init__(self, in_shape, in_channel, out_channel, nb_features=None, deep_supervision=True) -> None:
        super(FNestedUnet, self).__init__()

        self.deep_supervision = deep_supervision

        ndim = len(in_shape)
        assert ndim in [2, 3], 'the dim of the in_shape should be one of 2, or 3. found: %d' % ndim

        conv_fn = getattr(nn, "Conv{0}d".format(ndim))
        pool_fn = getattr(nn, "MaxPool{0}d".format(ndim))
        #self.upsample_fn = getattr(nn, "ConvTranspose{0}d".format(ndim))
        self.upsample_fn = nn.Upsample(scale_factor=2, mode='trilinear')

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = pool_fn(2)

        if nb_features is None:
            nb_features = [16, 32, 64, 128, 256]   # 512 is bottom neck layer

        self.conv0_0_0 = conv_fn(in_channel, nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_0_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_0 = ConvBlock(ndim, nb_features[0], nb_features[1])
        self.conv2_0 = ConvBlock(ndim, nb_features[1], nb_features[2])
        self.conv3_0 = ConvBlock(ndim, nb_features[2], nb_features[3])
        self.conv4_0 = ConvBlock(ndim, nb_features[3], nb_features[4])

        self.conv0_1_0 = conv_fn(nb_features[0] + nb_features[1], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_1_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_1 = ConvBlock(ndim, nb_features[1] + nb_features[2], nb_features[1])
        self.conv2_1 = ConvBlock(ndim, nb_features[2] + nb_features[3], nb_features[2])
        self.conv3_1 = ConvBlock(ndim, nb_features[3] + nb_features[4], nb_features[3])

        self.conv0_2 = ConvBlock(ndim, nb_features[0] * 2 + nb_features[1], nb_features[0])
        self.conv1_2 = ConvBlock(ndim, nb_features[1] * 2 + nb_features[2], nb_features[1])
        self.conv2_2 = ConvBlock(ndim, nb_features[2] * 2 + nb_features[3], nb_features[2])

        self.conv0_3 = ConvBlock(ndim, nb_features[0] * 3 + nb_features[1], nb_features[0])
        self.conv1_3 = ConvBlock(ndim, nb_features[1] * 3 + nb_features[2], nb_features[1])

        self.conv0_4 = ConvBlock(ndim, nb_features[0] * 4 + nb_features[1], nb_features[0])

        #self.up0_1 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        #self.up1_1 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        #self.up2_1 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)
        #self.up3_1 = upsample_fn(nb_features[4], nb_features[3], kernel_size=2, stride=2)

        #self.up0_2 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        #self.up1_2 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        #self.up2_2 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)

        #self.up0_3 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        #self.up1_3 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)

        #self.up0_4 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)

        if deep_supervision:
            self.final_1 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_2 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_3 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_4 = conv_fn(nb_features[0], out_channel, kernel_size=1)
        else:
            self.final = conv_fn(nb_features[0], out_channel, kernel_size=1)

    def forward(self, x):
        x0_0_0 = self.conv0_0_0(x)
        x0_0_1 = self.activation(x0_0_0)
        x0_0_2 = self.conv0_0_1(x0_0_1)
        x0_0_3 = self.activation(x0_0_2)
        x1_0 = self.conv1_0(self.maxpool(x0_0_3))
        x0_1_0 = self.conv0_1_0(torch.cat([x0_0_3, self.upsample_fn(x1_0)], dim=1))
        x0_1_1 = self.activation(x0_1_0)
        x0_1_2 = self.conv0_1_1(x0_1_1)
        x0_1_3 = self.activation(x0_1_2)

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample_fn(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0_3, x0_1_3, self.upsample_fn(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample_fn(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample_fn(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0_3, x0_1_3, x0_2, self.upsample_fn(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample_fn(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample_fn(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample_fn(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0_3, x0_1_3, x0_2, x0_3, self.upsample_fn(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final_1(x0_1_3)
            output2 = self.final_2(x0_2)
            output3 = self.final_3(x0_3)
            output4 = self.final_4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return [x0_0_0, x0_1_0, x0_2, x0_3, output]


class LFNestedUnet(nn.Module):
    def __init__(self, in_shape, in_channel, out_channel, nb_features=None, deep_supervision=True) -> None:
        super(LFNestedUnet, self).__init__()

        self.deep_supervision = deep_supervision

        ndim = len(in_shape)
        assert ndim in [2, 3], 'the dim of the in_shape should be one of 2, or 3. found: %d' % ndim

        conv_fn = getattr(nn, "Conv{0}d".format(ndim))
        pool_fn = getattr(nn, "MaxPool{0}d".format(ndim))
        upsample_fn = getattr(nn, "ConvTranspose{0}d".format(ndim))

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = pool_fn(2)

        if nb_features is None:
            nb_features = [32, 64, 128, 256, 512]   # 512 is bottom neck layer[16, 32, 64, 128, 256]

        self.conv0_0_0 = conv_fn(in_channel, nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_0_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_0 = ConvBlock(ndim, nb_features[0], nb_features[1])
        self.conv2_0 = ConvBlock(ndim, nb_features[1], nb_features[2])
        self.conv3_0 = ConvBlock(ndim, nb_features[2], nb_features[3])
        self.conv4_0 = ConvBlock(ndim, nb_features[3], nb_features[4])

        self.conv0_1_0 = conv_fn(nb_features[0] + nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_1_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_1 = ConvBlock(ndim, nb_features[1] + nb_features[1], nb_features[1])
        self.conv2_1 = ConvBlock(ndim, nb_features[2] + nb_features[2], nb_features[2])
        self.conv3_1 = ConvBlock(ndim, nb_features[3] + nb_features[3], nb_features[3])

        self.conv0_2 = ConvBlock(ndim, nb_features[0] * 2 + nb_features[0], nb_features[0])
        self.conv1_2 = ConvBlock(ndim, nb_features[1] * 2 + nb_features[1], nb_features[1])
        self.conv2_2 = ConvBlock(ndim, nb_features[2] * 2 + nb_features[2], nb_features[2])

        self.conv0_3 = ConvBlock(ndim, nb_features[0] * 3 + nb_features[0], nb_features[0])
        self.conv1_3 = ConvBlock(ndim, nb_features[1] * 3 + nb_features[1], nb_features[1])

        self.conv0_4 = ConvBlock(ndim, nb_features[0] * 4 + nb_features[0], nb_features[0])

        self.up0_1 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_1 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_1 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)
        self.up3_1 = upsample_fn(nb_features[4], nb_features[3], kernel_size=2, stride=2)

        self.up0_2 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_2 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_2 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)

        self.up0_3 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_3 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)

        self.up0_4 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)

        if deep_supervision:
            self.final_1 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_2 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_3 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_4 = conv_fn(nb_features[0], out_channel, kernel_size=1)
        else:
            self.final = conv_fn(nb_features[0], out_channel, kernel_size=1)

    def forward(self, x):
        x0_0_0 = self.conv0_0_0(x)
        x0_0_1 = self.activation(x0_0_0)
        x0_0_2 = self.conv0_0_1(x0_0_1)
        x0_0_3 = self.activation(x0_0_2)
        x1_0 = self.conv1_0(self.maxpool(x0_0_3))
        x0_1_0 = self.conv0_1_0(torch.cat([x0_0_3, self.up0_1(x1_0)], dim=1))
        x0_1_1 = self.activation(x0_1_0)
        x0_1_2 = self.conv0_1_1(x0_1_1)
        x0_1_3 = self.activation(x0_1_2)

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0_3, x0_1_3, self.up0_2(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1_2(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0_3, x0_1_3, x0_2, self.up0_3(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1_3(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0_3, x0_1_3, x0_2, x0_3, self.up0_4(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final_1(x0_1_3)
            output2 = self.final_2(x0_2)
            output3 = self.final_3(x0_3)
            output4 = self.final_4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return [x0_0_0, x0_1_0, x0_2, x0_3, output]


class SLFNestedUnet(nn.Module):
    def __init__(self, in_shape, in_channel, out_channel, nb_features=None, deep_supervision=True) -> None:
        super(SLFNestedUnet, self).__init__()

        self.deep_supervision = deep_supervision

        ndim = len(in_shape)
        assert ndim in [2, 3], 'the dim of the in_shape should be one of 2, or 3. found: %d' % ndim

        conv_fn = getattr(nn, "Conv{0}d".format(ndim))
        pool_fn = getattr(nn, "MaxPool{0}d".format(ndim))
        upsample_fn = getattr(nn, "ConvTranspose{0}d".format(ndim))

        self.maxpool1 = nn.MaxPool3d(kernel_size=[2,2,1], stride=[2,2,1])

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = pool_fn(2)

        if nb_features is None:
            nb_features = [16, 32, 64, 128, 256]   # 512 is bottom neck layer

        self.conv0_0_0 = conv_fn(in_channel, nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_0_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_0 = ConvBlock(ndim, nb_features[0], nb_features[1])
        self.conv2_0 = ConvBlock(ndim, nb_features[1], nb_features[2])
        self.conv3_0 = ConvBlock(ndim, nb_features[2], nb_features[3])
        self.conv4_0 = ConvBlock(ndim, nb_features[3], nb_features[4])

        self.conv0_1_0 = conv_fn(nb_features[0] + nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_1_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_1 = ConvBlock(ndim, nb_features[1] + nb_features[1], nb_features[1])
        self.conv2_1 = ConvBlock(ndim, nb_features[2] + nb_features[2], nb_features[2])
        self.conv3_1 = ConvBlock(ndim, nb_features[3] + nb_features[3], nb_features[3])

        self.conv0_2 = ConvBlock(ndim, nb_features[0] * 2 + nb_features[0], nb_features[0])
        self.conv1_2 = ConvBlock(ndim, nb_features[1] * 2 + nb_features[1], nb_features[1])
        self.conv2_2 = ConvBlock(ndim, nb_features[2] * 2 + nb_features[2], nb_features[2])

        self.conv0_3 = ConvBlock(ndim, nb_features[0] * 3 + nb_features[0], nb_features[0])
        self.conv1_3 = ConvBlock(ndim, nb_features[1] * 3 + nb_features[1], nb_features[1])

        self.conv0_4 = ConvBlock(ndim, nb_features[0] * 4 + nb_features[0], nb_features[0])

        self.up0_1 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_1 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_1 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)
        self.up3_1 = upsample_fn(nb_features[4], nb_features[3], kernel_size=[2,2,1], stride=[2,2,1])

        self.up0_2 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_2 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_2 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)

        self.up0_3 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_3 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)

        self.up0_4 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)

        if deep_supervision:
            self.final_1 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_2 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_3 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            self.final_4 = conv_fn(nb_features[0], out_channel, kernel_size=1)
        else:
            self.final = conv_fn(nb_features[0], out_channel, kernel_size=1)

    def forward(self, x):
        x0_0_0 = self.conv0_0_0(x)
        x0_0_1 = self.activation(x0_0_0)
        x0_0_2 = self.conv0_0_1(x0_0_1)
        x0_0_3 = self.activation(x0_0_2)
        x1_0 = self.conv1_0(self.maxpool(x0_0_3))
        x0_1_0 = self.conv0_1_0(torch.cat([x0_0_3, self.up0_1(x1_0)], dim=1))
        x0_1_1 = self.activation(x0_1_0)
        x0_1_2 = self.conv0_1_1(x0_1_1)
        x0_1_3 = self.activation(x0_1_2)

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0_3, x0_1_3, self.up0_2(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1_2(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0_3, x0_1_3, x0_2, self.up0_3(x1_2)], dim=1))
        x_pool = self.maxpool1(x3_0)
        x4_0 = self.conv4_0(self.maxpool1(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1_3(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0_3, x0_1_3, x0_2, x0_3, self.up0_4(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final_1(x0_1_3)
            output2 = self.final_2(x0_2)
            output3 = self.final_3(x0_3)
            output4 = self.final_4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return [x0_0_0, x0_1_0, x0_2, x0_3, output]


class DLFNestedUnet(nn.Module):
    def __init__(self, in_shape, in_channel, out_channel, nb_features=None, deep_supervision=True) -> None:
        super(DLFNestedUnet, self).__init__()

        self.deep_supervision = deep_supervision

        ndim = len(in_shape)
        assert ndim in [2, 3], 'the dim of the in_shape should be one of 2, or 3. found: %d' % ndim

        conv_fn = getattr(nn, "Conv{0}d".format(ndim))
        pool_fn = getattr(nn, "MaxPool{0}d".format(ndim))
        upsample_fn = getattr(nn, "ConvTranspose{0}d".format(ndim))

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = pool_fn(2)

        if nb_features is None:
            nb_features = [16, 32, 64, 128]   # 512 is bottom neck layer[16, 32, 64, 128, 256]

        self.conv0_0_0 = conv_fn(in_channel, nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_0_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_0 = ConvBlock(ndim, nb_features[0], nb_features[1])
        self.conv2_0 = ConvBlock(ndim, nb_features[1], nb_features[2])
        self.conv3_0 = ConvBlock(ndim, nb_features[2], nb_features[3])
        #self.conv4_0 = ConvBlock(ndim, nb_features[3], nb_features[4])

        self.conv0_1_0 = conv_fn(nb_features[0] + nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv0_1_1 = conv_fn(nb_features[0], nb_features[0], kernel_size=3, stride=1, padding=1)
        self.conv1_1 = ConvBlock(ndim, nb_features[1] + nb_features[1], nb_features[1])
        self.conv2_1 = ConvBlock(ndim, nb_features[2] + nb_features[2], nb_features[2])
        #self.conv3_1 = ConvBlock(ndim, nb_features[3] + nb_features[3], nb_features[3])

        self.conv0_2 = ConvBlock(ndim, nb_features[0] * 2 + nb_features[0], nb_features[0])
        self.conv1_2 = ConvBlock(ndim, nb_features[1] * 2 + nb_features[1], nb_features[1])
        #self.conv2_2 = ConvBlock(ndim, nb_features[2] * 2 + nb_features[2], nb_features[2])

        self.conv0_3 = ConvBlock(ndim, nb_features[0] * 3 + nb_features[0], nb_features[0])
        #self.conv1_3 = ConvBlock(ndim, nb_features[1] * 3 + nb_features[1], nb_features[1])

        #self.conv0_4 = ConvBlock(ndim, nb_features[0] * 4 + nb_features[0], nb_features[0])

        self.up0_1 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_1 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        self.up2_1 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)
        #self.up3_1 = upsample_fn(nb_features[4], nb_features[3], kernel_size=2, stride=2)

        self.up0_2 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        self.up1_2 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)
        #self.up2_2 = upsample_fn(nb_features[3], nb_features[2], kernel_size=2, stride=2)

        self.up0_3 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)
        #self.up1_3 = upsample_fn(nb_features[2], nb_features[1], kernel_size=2, stride=2)

        #self.up0_4 = upsample_fn(nb_features[1], nb_features[0], kernel_size=2, stride=2)

        if deep_supervision:
            self.final_1 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            #self.final_2 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            #self.final_3 = conv_fn(nb_features[0], out_channel, kernel_size=1)
            #self.final_4 = conv_fn(nb_features[0], out_channel, kernel_size=1)
        else:
            self.final = conv_fn(nb_features[0], out_channel, kernel_size=1)

    def forward(self, x):
        x0_0_0 = self.conv0_0_0(x)
        x0_0_1 = self.activation(x0_0_0)
        x0_0_2 = self.conv0_0_1(x0_0_1)
        x0_0_3 = self.activation(x0_0_2)
        x1_0 = self.conv1_0(self.maxpool(x0_0_3))
        x0_1_0 = self.conv0_1_0(torch.cat([x0_0_3, self.up0_1(x1_0)], dim=1))
        x0_1_1 = self.activation(x0_1_0)
        x0_1_2 = self.conv0_1_1(x0_1_1)
        x0_1_3 = self.activation(x0_1_2)

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1_1(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0_3, x0_1_3, self.up0_2(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2_1(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1_2(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0_3, x0_1_3, x0_2, self.up0_3(x1_2)], dim=1))

        #x4_0 = self.conv4_0(self.maxpool(x3_0))

        #x3_1 = self.conv3_1(torch.cat([x3_0, self.up3_1(x4_0)], dim=1))
        #x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2_2(x3_1)], dim=1))
        #x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1_3(x2_2)], dim=1))
        #x0_4 = self.conv0_4(torch.cat([x0_0_3, x0_1_3, x0_2, x0_3, self.up0_4(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final_1(x0_3)
            #output2 = self.final_2(x0_2)
            #output3 = self.final_3(x0_3)
            #output4 = self.final_4(x0_4)
            return [output1]

        else:
            output = self.final(x0_3)
            return [x0_0_0, x0_1_0, output]