import torch.nn as nn
import torch
from torch.nn import init
from networks.util import ResnetBlock
import copy


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
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  # nn.InstanceNorm2d(32, eps=1e-5),
                                  nn.LeakyReLU(0.3),

                                  nn.Conv2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  # nn.InstanceNorm2d(32, eps=1e-5),
                                  nn.LeakyReLU(0.3),
                                  # nn.RReLU(lower=0.1, upper=0.2),

                                  nn.Conv2d(in_channels=32,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  # nn.InstanceNorm2d(64, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.3),
                                  # nn.Dropout(0.5)
                                  )
        init_weight_(self.net1)

        self.net2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=128,
                                            kernel_size=5,
                                            stride=2,
                                            padding=2,
                                            bias=False),
                                  nn.InstanceNorm2d(128, eps=1e-5),
                                  nn.LeakyReLU(0.3),
                                  # nn.RReLU(lower=0.1, upper=0.2),

                                  nn.Conv2d(in_channels=128,
                                            out_channels=256,
                                            kernel_size=5,
                                            stride=2,
                                            padding=2,
                                            bias=False),
                                  nn.InstanceNorm2d(256, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.3),
                                  # nn.Dropout(0.5),
                                  )
        init_weight_(self.net2)

        self.net3 = nn.Sequential(nn.Conv2d(in_channels=256,
                                            out_channels=1,
                                            kernel_size=7,
                                            stride=1,
                                            padding=3,
                                            bias=False)
                                  )
        init_weight_(self.net3)

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)

        x = self.net1(x)
        # (batch * 64 * 16 * 21)
        # ↓
        # (batch * 256 * 16 * 21)

        x = self.net2(x)
        # (batch * 1 * 16 * 21)
        x = self.net3(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.paragraph_cnet1 = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                             nn.Conv2d(in_channels=1,
                                                       out_channels=32,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False),
                                             nn.InstanceNorm2d(32, eps=1e-5),
                                             nn.LeakyReLU(0.3),
                                             # nn.Dropout(0.5),

                                             nn.ReflectionPad2d((1, 1, 1, 1)),
                                             nn.Conv2d(in_channels=32,
                                                       out_channels=32,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False),
                                             nn.InstanceNorm2d(32, eps=1e-5),
                                             nn.LeakyReLU(0.3),
                                             # nn.Dropout(0.5),

                                             nn.ReflectionPad2d((1, 1, 1, 1)),
                                             nn.Conv2d(in_channels=32,
                                                       out_channels=64,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False),
                                             nn.InstanceNorm2d(64, eps=1e-5),
                                             nn.ReLU(),
                                             # nn.Dropout(0.5)
                                             )
        init_weight_(self.paragraph_cnet1)

        self.bar_cnet1 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                 out_channels=128,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(128, eps=1e-5),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),


                                       nn.Conv2d(in_channels=128,
                                                 out_channels=64,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(64, eps=1e-5),
                                       nn.ReLU()
                                       )

        self.bar_cnet2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                 out_channels=128,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(128, eps=1e-5),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),


                                       nn.Conv2d(in_channels=128,
                                                 out_channels=64,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(64, eps=1e-5),
                                       nn.ReLU()
                                       )

        self.bar_cnet3 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                 out_channels=128,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(128, eps=1e-5),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),


                                       nn.Conv2d(in_channels=128,
                                                 out_channels=64,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(64, eps=1e-5),
                                       nn.ReLU()
                                       )

        self.bar_cnet4 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                 out_channels=128,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(128, eps=1e-5),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),


                                       nn.Conv2d(in_channels=128,
                                                 out_channels=64,
                                                 kernel_size=5,
                                                 stride=1,
                                                 padding=2,
                                                 bias=False),
                                       nn.InstanceNorm2d(64, eps=1e-5),
                                       nn.ReLU()
                                       )

        init_weight_(self.bar_cnet1)
        init_weight_(self.bar_cnet2)
        init_weight_(self.bar_cnet3)
        init_weight_(self.bar_cnet4)

        self.paragraph_cnet2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                       out_channels=128,
                                                       kernel_size=5,
                                                       stride=2,
                                                       padding=2,
                                                       bias=False),
                                             nn.InstanceNorm2d(128, eps=1e-5),
                                             nn.ReLU(),
                                             # nn.Dropout(0.5),

                                             nn.Conv2d(in_channels=128,
                                                       out_channels=256,
                                                       kernel_size=5,
                                                       stride=2,
                                                       padding=2,
                                                       bias=False),
                                             nn.InstanceNorm2d(256, eps=1e-5),
                                             nn.ReLU(),
                                             # nn.Dropout(0.5)
                                             )

        init_weight_(self.paragraph_cnet2)

        self.resnet = nn.Sequential()
        for i in range(10):
            self.resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                               padding_type='reflect',
                                                               use_dropout=False,
                                                               use_bias=False,
                                                               norm_layer=nn.InstanceNorm2d))
        init_weight_(self.resnet)

        self.paragraph_ctnet1 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,
                                                                 out_channels=128,
                                                                 kernel_size=5,
                                                                 stride=2,
                                                                 padding=2,
                                                                 bias=False),
                                              nn.ZeroPad2d((0, 1, 0, 1)),
                                              nn.InstanceNorm2d(128, eps=1e-5),
                                              nn.ReLU(),
                                              # nn.Dropout(0.5),

                                              nn.ConvTranspose2d(in_channels=128,
                                                                 out_channels=64,
                                                                 kernel_size=5,
                                                                 stride=2,
                                                                 padding=2,
                                                                 bias=False),
                                              nn.ZeroPad2d((1, 0, 1, 0)),
                                              nn.InstanceNorm2d(64, eps=1e-5),
                                              nn.ReLU()
                                              )
        init_weight_(self.paragraph_ctnet1)
        '''
        self.bar_ctnet = nn.Sequential(nn.ConvTranspose2d(in_channels=64,
                                                          out_channels=128,
                                                          kernel_size=3,
                                                          stride=1,
                                                          padding=1,
                                                          bias=False),
                                       nn.InstanceNorm2d(128, eps=1e-5),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),

                                       nn.ConvTranspose2d(in_channels=128,
                                                          out_channels=64,
                                                          kernel_size=3,
                                                          stride=1,
                                                          padding=1,
                                                          bias=False),
                                       nn.InstanceNorm2d(64, eps=1e-5),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5)
                                       )
        init_weight_(self.bar_ctnet)
        '''

        self.paragraph_cnet3 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                                             nn.Conv2d(in_channels=64,
                                                       out_channels=1,
                                                       kernel_size=7,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False),
                                             nn.Sigmoid()
                                             )
        init_weight_(self.paragraph_cnet3)

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)

        x = self.paragraph_cnet1(x)
        # (batch * 1 * 70 * 90) after padded
        # ↓
        # (batch * 64 * 64 * 84)
        # ↓
        # (batch * 256 * 32 * 42)

        x1, x2, x3, x4 = x.split([16, 16, 16, 16], dim=2)

        # (batch * 64 * 32 * 42)

        x1 = self.bar_cnet1(x1)
        x2 = self.bar_cnet2(x2)
        x3 = self.bar_cnet3(x3)
        x4 = self.bar_cnet4(x4)

        # (batch * 64 * 16 * 21)

        x = torch.cat([x1, x2, x3, x4], dim=2)

        # x = self.after_bar(x)
        '''
        del x1
        del x2
        del x3
        del x4
        '''

        x = self.paragraph_cnet2(x)
        # (batch * 256 * 16 * 21)

        x = self.resnet(x)

        # (batch * 256 * 16 * 21)

        x = self.paragraph_ctnet1(x)
        # (batch * 256 * 32 * 42)
        # ↓
        # (batch * 64 * 64 * 84)
        # ↓
        # After padding, (batch * 64 * 70 * 90)
        # ↓
        x = self.paragraph_cnet3(x)
        # (batch * 1 * 64 * 84)

        return x

