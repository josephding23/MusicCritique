import torch.nn as nn
import torch
from torch.nn import init
from networks.util import ResnetBlock


def init_weight_(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.02)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # shape = (64, 84, 1)
        # df_dim = 64

        self.net1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            out_channels=64,
                                            kernel_size=7,
                                            stride=1,
                                            padding=3,
                                            bias=False),
                                  nn.InstanceNorm2d(64, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.4),
                                  nn.Dropout(0.8),

                                  nn.Conv2d(in_channels=64,
                                            out_channels=256,
                                            kernel_size=7,
                                            stride=2,
                                            padding=3,
                                            bias=False),
                                  nn.InstanceNorm2d(256, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.4),
                                  nn.Dropout(0.8),
                                  )
        init_weight_(self.net1)

        self.net2 = nn.Sequential(nn.Conv2d(in_channels=256,
                                            out_channels=1,
                                            kernel_size=7,
                                            stride=1,
                                            padding=3,
                                            bias=False),
                                  )
        init_weight_(self.net2)

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)

        x = self.net1(x)
        # (batch * 16 * 64 * 84)
        # ↓
        # (batch * 64 * 16 * 21)
        # ↓
        # (batch * 256 * 16 * 21)

        x = self.net2(x)
        # (batch * 1 * 16 * 21)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.paragraph_net1 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                                            nn.Conv2d(in_channels=1,
                                                      out_channels=64,
                                                      kernel_size=7,
                                                      stride=1,
                                                      padding=0,
                                                      bias=False),
                                            nn.InstanceNorm2d(64, eps=1e-5),
                                            nn.LeakyReLU(negative_slope=0.2),

                                            nn.Conv2d(in_channels=64,
                                                      out_channels=128,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      bias=False),
                                            nn.InstanceNorm2d(128, eps=1e-5),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            )
        init_weight_(self.paragraph_net1)

        self.bar_cnet = nn.Sequential(nn.Conv2d(in_channels=32,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                bias=False),
                                      nn.InstanceNorm2d(64, eps=1e-5),
                                      nn.LeakyReLU(negative_slope=0.2))
        init_weight_(self.bar_cnet)

        self.resnet = nn.Sequential()
        for i in range(10):
            self.resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                               padding_type='reflect',
                                                               use_dropout=True,
                                                               use_bias=False,
                                                               norm_layer=nn.InstanceNorm2d))

        self.bar_tcnet = nn.Sequential(nn.ConvTranspose2d(in_channels=64,
                                                          out_channels=32,
                                                          kernel_size=3,
                                                          stride=2,
                                                          padding=1,
                                                          bias=False),
                                       nn.ZeroPad2d((0, 1, 0, 1)),
                                       nn.InstanceNorm2d(32, eps=1e-5),
                                       nn.LeakyReLU(negative_slope=0.2))

        init_weight_(self.bar_tcnet)

        self.paragraph_net2 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,
                                                               out_channels=64,
                                                               kernel_size=3,
                                                               stride=2,
                                                               padding=1,
                                                               bias=False),
                                            nn.ZeroPad2d((0, 1, 0, 1)),
                                            nn.InstanceNorm2d(64, eps=1e-5),
                                            nn.LeakyReLU(negative_slope=0.2),

                                            nn.ReflectionPad2d((3, 3, 3, 3)),
                                            nn.Conv2d(in_channels=64,
                                                      out_channels=1,
                                                      kernel_size=7,
                                                      stride=1,
                                                      padding=0,
                                                      bias=False),
                                            # nn.Sigmoid()
                                            )
        init_weight_(self.paragraph_net2)

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)

        x = self.paragraph_net1(x)
        # (batch * 1 * 70 * 90) after padded
        # ↓
        # (batch * 64 * 64 * 84)
        # ↓
        # (batch * 256 * 32 * 42)

        x1, x2, x3, x4 = x.split([32, 32, 32, 32], dim=1)

        # (batch * 64 * 32 * 42)
        x1 = self.bar_cnet(x1)
        x2 = self.bar_cnet(x2)
        x3 = self.bar_cnet(x3)
        x4 = self.bar_cnet(x4)
        # (batch * 64 * 16 * 21)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        # (batch * 256 * 16 * 21)

        x = self.resnet(x)

        # (batch * 256 * 16 * 21)

        x1, x2, x3, x4 = x.split([64, 64, 64, 64], dim=1)

        # (batch * 64 * 16 * 21)
        x1 = self.bar_tcnet(x1)
        x2 = self.bar_tcnet(x2)
        x3 = self.bar_tcnet(x3)
        x4 = self.bar_tcnet(x4)
        # (batch * 64 * 32 * 42)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.paragraph_net2(x)
        # (batch * 256 * 32 * 42)
        # ↓
        # (batch * 64 * 64 * 84)
        # ↓
        # After padding, (batch * 64 * 70 * 90)
        # ↓
        # (batch * 1 * 64 * 84)

        return x
