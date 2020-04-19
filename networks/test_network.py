import torch.nn as nn
import torch
from torch.nn import init
from networks.util import ResnetBlock


def test_G():
    net1 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                                            nn.Conv2d(in_channels=1,
                                                      out_channels=32,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=0,
                                                      bias=False),
                                            # nn.InstanceNorm2d(64, eps=1e-5),
                                            nn.LeakyReLU(0.2))

    net2 = nn.Sequential(
                        nn.Conv2d(in_channels=32,
                                                      out_channels=32,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=0,
                                                      bias=False),
                                            # nn.ReflectionPad2d((0, 1, 0, 1)),
                                            nn.InstanceNorm2d(64, eps=1e-5),
                                            nn.LeakyReLU(0.2)
                                            # nn.RReLU(lower=0.2, upper=0.4),
                                            )
    net3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                                      out_channels=64,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=0,
                                                      bias=False),
                                            # nn.ReflectionPad2d((0, 1, 0, 1)),
                                            nn.InstanceNorm2d(64, eps=1e-5),
                                            nn.LeakyReLU(0.2))
                                            # nn.RReLU(lower=0.2, upper=0.4),.Conv2d)
    bar_cnet1 = nn.Sequential(nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2,
                                    bias=False),
                          # nn.InstanceNorm2d(128, eps=1e-5),
                          nn.LeakyReLU(0.2),
                          nn.Dropout(0.5))

    bar_cnet2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                        out_channels=256,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        bias=False),
                              # nn.InstanceNorm2d(128, eps=1e-5),
                              nn.LeakyReLU(0.2),
                              nn.Dropout(0.5))

    bar_cnet3 = nn.Sequential(nn.Conv2d(in_channels=256,
                                        out_channels=128,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        bias=False),
                              # nn.InstanceNorm2d(128, eps=1e-5),
                              nn.LeakyReLU(0.2),
                              nn.Dropout(0.5))

    bar_cnet4 = nn.Sequential(nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2,
                                    bias=False),
                          nn.InstanceNorm2d(64, eps=1e-5),
                          nn.LeakyReLU(0.2),
                          nn.Dropout(0.5)
                          )

    net4 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                      out_channels=128,
                                                      kernel_size=5,
                                                      stride=2,
                                                      padding=2,
                                                      bias=False),
                                            nn.InstanceNorm2d(128, eps=1e-5),
                                            nn.LeakyReLU(0.2)
                                        )


    net6 = nn.Sequential(nn.Conv2d(in_channels=128,
              out_channels=256,
              kernel_size=5,
              stride=2,
              padding=2,
              bias=False),
    nn.InstanceNorm2d(256, eps=1e-5),
    nn.ReLU())


    resnet = nn.Sequential()
    for i in range(10):
        resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                           padding_type='reflect',
                                                           use_dropout=False,
                                                           use_bias=False,
                                                           norm_layer=nn.InstanceNorm2d))


    net7 = nn.Sequential(

                        nn.ConvTranspose2d(in_channels=256,
                                                               out_channels=128,
                                                               kernel_size=5,
                                                               stride=2,
                                                               padding=2,
                                                               bias=False),
                        nn.ZeroPad2d((0, 1, 0, 1)),
                                            nn.InstanceNorm2d(128, eps=1e-5),
                                            nn.LeakyReLU(0.2))


    net8 = nn.Sequential(

                        nn.ConvTranspose2d(in_channels=128,
                                                               out_channels=128,
                                                               kernel_size=5,
                                                               stride=1,
                                                               padding=2,
                                                               bias=False),
                        # nn.ZeroPad2d((0, 1, 0, 1)),
                                            nn.InstanceNorm2d(128, eps=1e-5),
                                            nn.LeakyReLU(0.2))

    net9 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=128,
                                                               out_channels=64,
                                                               kernel_size=3,
                                                               stride=2,
                                                               padding=1,
                                                               bias=False),
    nn.ZeroPad2d((1, 0, 1, 0)),
                                            nn.InstanceNorm2d(64, eps=1e-5),
                                            nn.LeakyReLU(0.2))

    net10 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                         nn.Conv2d(in_channels=64,
                                   out_channels=1,
                                   kernel_size=7,
                                   stride=1,
                                   padding=0,
                                   bias=False))


    x = torch.zeros((1, 1, 64, 84))
    # print(x.shape)

    x = net1(x)
    # print(x.shape)

    x = net2(x)
    # print(x.shape)

    x = net3(x)
    # print(x.shape)



    x1, x2, x3, x4 = x.split([16, 16, 16, 16], dim=2)
    print(x1.shape)

    x1 = bar_cnet1(x1)
    print(x1.shape)

    x1 = bar_cnet2(x1)
    print(x1.shape)

    x1 = bar_cnet3(x1)
    print(x1.shape)

    x1 = bar_cnet4(x1)
    print(x1.shape)




    
    x = net4(x)
    print(x.shape)


    x = net6(x)
    print(x.shape)
    
    
    x = resnet(x)
    print(x.shape)



    x = net7(x)
    print(x.shape)

    x = net8(x)
    print(x.shape)

    x = net9(x)
    print(x.shape)

    x = net10(x)
    print(x.shape)

    # x = net9(x)
    # print(x.shape)

    # x = net10(x)
    # print(x.shape)


def test_D():
    net11 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  # nn.InstanceNorm2d(16, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.3))

    net12 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  # nn.InstanceNorm2d(16, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.3))

    net13 = nn.Sequential(nn.Conv2d(in_channels=32,
                                            out_channels=64,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=False),
                                  nn.InstanceNorm2d(64, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.3),
                                  nn.Dropout(0.5))

    net21 = nn.Sequential(nn.Conv2d(in_channels=64,
                                            out_channels=128,
                                            kernel_size=5,
                                            stride=2,
                                            padding=2,
                                            bias=False),
                                  # nn.InstanceNorm2d(128, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.3))


    net23 = nn.Sequential(nn.Conv2d(in_channels=128,
                                            out_channels=256,
                                            kernel_size=5,
                                            stride=2,
                                            padding=2,
                                            bias=False),
                                  nn.InstanceNorm2d(256, eps=1e-5),
                                  nn.RReLU(lower=0.2, upper=0.3))

    net3 = nn.Sequential(nn.Conv2d(in_channels=256,
                                            out_channels=1,
                                            kernel_size=7,
                                            stride=1,
                                            padding=3,
                                            bias=False
                                  ))
    net31 = nn.Sequential(nn.Conv2d(in_channels=256,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=[1, 3],
                                    padding=1,
                                    bias=False),
                          # nn.InstanceNorm2d(128, eps=1e-5),
                          nn.RReLU(lower=0.2, upper=0.3))

    net32 = nn.Sequential(nn.Conv2d(in_channels=64,
                                    out_channels=16,
                                    kernel_size=3,
                                    stride=[2, 1],
                                    padding=1,
                                    bias=False),
                          # nn.InstanceNorm2d(128, eps=1e-5),
                          nn.RReLU(lower=0.2, upper=0.3))

    net33 = nn.Sequential(nn.Conv2d(in_channels=16,
                                    out_channels=1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False),
                          nn.InstanceNorm2d(256, eps=1e-5),
                          nn.RReLU(lower=0.2, upper=0.3))


    x = torch.zeros((1, 1, 64, 84))

    x = net11(x)
    print(x.shape)

    x = net12(x)
    print(x.shape)

    x = net13(x)
    print(x.shape)

    x = net21(x)
    print(x.shape)

    x = net23(x)
    print(x.shape)

    x = net3(x)
    print(x.shape)

    # x = net31(x)
    # print(x.shape)

    # x = net32(x)
    # print(x.shape)

    # x = net33(x)
    # print(x.shape)



if __name__ == '__main__':
    test_D()
