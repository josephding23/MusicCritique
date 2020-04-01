import torch.nn as nn
import torch
from torch.nn import init
from networks.util import ResnetBlock


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # shape = (64, 84, 1)
        # df_dim = 64

        conv1 = nn.Conv2d(in_channels=1,
                          out_channels=16,
                          kernel_size=7,
                          stride=1,
                          padding=3,
                          bias=False)
        init.normal_(conv1.weight, mean=0.0, std=0.02)

        conv2 = nn.Conv2d(in_channels=16,
                          out_channels=64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        init.normal_(conv2.weight, mean=0.0, std=0.02)

        conv3 = nn.Conv2d(in_channels=64,
                          out_channels=256,
                          kernel_size=7,
                          stride=1,
                          padding=3,
                          bias=False)
        init.normal_(conv3.weight, mean=0.0, std=0.02)
        self.net1 = nn.Sequential(conv1,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  conv2,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  conv3,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.5)
                                  )

        conv4 = nn.Conv2d(in_channels=64,
                          out_channels=16,
                          kernel_size=7,
                          stride=1,
                          padding=3,
                          bias=False)
        conv5 = nn.Conv2d(in_channels=16,
                          out_channels=1,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        init.normal_(conv4.weight, mean=0.0, std=0.02)
        self.net2 = nn.Sequential(conv4,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  conv5,
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Dropout(0.5)
                                  )

        conv6 = nn.Conv2d(in_channels=4,
                          out_channels=1,
                          kernel_size=7,
                          stride=1,
                          padding=3,
                          bias=False)
        init.normal_(conv6.weight, mean=0.0, std=0.02)
        self.net3 = nn.Sequential(conv6)

    def forward(self, tensor_in):
        x = tensor_in
        x = self.net1(x)

        x1, x2, x3, x4 = x.split([64, 64, 64, 64], dim=1)

        x1 = self.net2(x1)
        x2 = self.net2(x2)
        x3 = self.net2(x3)
        x4 = self.net2(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.net3(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        conv1 = nn.Conv2d(in_channels=1,
                          out_channels=64,
                          kernel_size=7,
                          stride=1,
                          padding=0,
                          bias=False)
        init.normal_(conv1.weight, mean=0, std=0.02)
        self.cnet1 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                                   conv1,
                                   nn.InstanceNorm2d(64, eps=1e-5),
                                   nn.ReLU(),
                                   nn.Dropout(0.5))

        conv2 = nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False)
        init.normal_(conv2.weight, mean=0, std=0.02)
        self.cnet2 = nn.Sequential(conv2,
                                   nn.InstanceNorm2d(64, eps=1e-5),
                                   nn.ReLU(),
                                   nn.Dropout(0.5))

        conv3 = nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False)
        init.normal_(conv3.weight, mean=0, std=0.02)
        self.cnet3 = nn.Sequential(conv3,
                                   nn.InstanceNorm2d(128, eps=1e-5),
                                   nn.ReLU())

        self.resnet = nn.Sequential()
        for i in range(20):
            self.resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                               padding_type='reflect',
                                                               use_dropout=False,
                                                               use_bias=False,
                                                               norm_layer=nn.InstanceNorm2d))

        tconv1 = nn.ConvTranspose2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False)

        init.normal_(tconv1.weight, mean=0, std=0.02)
        self.tcnet1 = nn.Sequential(tconv1,
                                    nn.ZeroPad2d((0, 1, 0, 1)),
                                    nn.InstanceNorm2d(256, eps=1e-5),
                                    nn.ReLU())

        tconv2 = nn.ConvTranspose2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                     bias=False)

        init.normal_(tconv2.weight, mean=0, std=0.02)
        self.tcnet2 = nn.Sequential(tconv2,
                                    nn.ZeroPad2d((0, 1, 0, 1)),
                                    nn.InstanceNorm2d(256, eps=1e-5),
                                    nn.ReLU())

        conv4 = nn.Conv2d(in_channels=64,
                          out_channels=1,
                          kernel_size=7,
                          stride=1,
                          padding=0)
        init.normal_(conv4.weight, mean=0, std=0.02)

        self.cnet4 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                                   conv4,
                                   nn.Sigmoid())

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)

        x = self.cnet1(x)
        # (batch * 1 * 70 * 90) after padded
        # (batch * 64 * 64 * 84)
        x = self.cnet2(x)
        # (batch * 128 * 32 * 42)

        x1, x2, x3, x4 = x.split([32, 32, 32, 32], dim=1)

        x1 = self.cnet3(x1)
        x2 = self.cnet3(x2)
        x3 = self.cnet3(x3)
        x4 = self.cnet3(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        # x = self.cnet3(x)
        # (batch * 256 * 16 * 21)

        x = self.resnet(x)
        # (batch * 256 * 16 * 21)

        x1, x2, x3, x4 = x.split([64, 64, 64, 64], dim=1)

        x1 = self.tcnet1(x1)
        x2 = self.tcnet1(x2)
        x3 = self.tcnet1(x3)
        x4 = self.tcnet1(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        # x = self.tcnet1(x)
        # (batch * 128 * 32 * 42)
        x = self.tcnet2(x)
        # (batch * 64 * 64 * 84)
        x = self.cnet4(x)
        # After padding, (batch * 64 * 70 * 90)
        # (batch * 1 * 64 * 84)

        return x
