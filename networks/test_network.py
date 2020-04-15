import torch.nn as nn
import torch
from torch.nn import init
from networks.util import ResnetBlock


def test_deeper_d():
    # shape = (64, 84, 1)
    # df_dim = 64

    # As a whole
    paragraph_net1 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                                        nn.Conv2d(in_channels=1,
                                                  out_channels=64,
                                                  kernel_size=7,
                                                  stride=1,
                                                  padding=0,
                                                  bias=False),
                                        nn.InstanceNorm2d(64, eps=1e-5),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        # nn.RReLU(lower=0.2, upper=0.4),

                                        )

    bar_cnet = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=128,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False),
                                  nn.InstanceNorm2d(128, eps=1e-5),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(in_channels=128,
                                            out_channels=256,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            bias=False),
                                  nn.InstanceNorm2d(256, eps=1e-5),
                                  nn.LeakyReLU(negative_slope=0.2)
                                  )

    bar_cnet_after = nn.Sequential(nn.InstanceNorm2d(256, eps=1e-5),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        )


    resnet = nn.Sequential()
    for i in range(10):
        resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                           padding_type='reflect',
                                                           use_dropout=True,
                                                           use_bias=False,
                                                           norm_layer=nn.InstanceNorm2d))


    paragraph_net2 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,
                                                           out_channels=128,
                                                           kernel_size=3,
                                                           stride=2,
                                                           padding=1,
                                                           bias=False),
                                        nn.ZeroPad2d((0, 1, 0, 1)),
                                        nn.InstanceNorm2d(128, eps=1e-5),
                                        nn.LeakyReLU(negative_slope=0.2),

                                        nn.ConvTranspose2d(in_channels=128,
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
    x = torch.ones(1, 1, 64, 84)
    print(x.shape)

    x = paragraph_net1(x)
    print(x.shape)

    x1, x2, x3, x4 = x.split([16, 16, 16, 16], dim=2)
    print(x1.shape)

    x1 = bar_cnet(x1)
    x2 = bar_cnet(x2)
    x3 = bar_cnet(x3)
    x4 = bar_cnet(x4)

    x = torch.cat([x1, x2, x3, x4], dim=2)
    x = bar_cnet_after(x)

    print(x.shape)

    x = resnet(x)
    print(x.shape)

    x = paragraph_net2(x)
    print(x.shape)

    # x = net4(x)
    # print(x.shape)



if __name__ == '__main__':
    test_deeper_d()
