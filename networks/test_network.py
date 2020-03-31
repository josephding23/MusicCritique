import torch.nn as nn
import torch
from torch.nn import init
from networks.util import ResnetBlock


def test_deeper_d():
    # shape = (64, 84, 1)
    # df_dim = 64

    # As a whole
    conv1 = nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False)
    init.normal_(conv1.weight, mean=0.0, std=0.02)
    net1 = nn.Sequential(conv1, nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.5))

    conv2 = nn.Conv2d(in_channels=16,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False)
    init.normal_(conv2.weight, mean=0.0, std=0.02)
    net2 = nn.Sequential(conv2, nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.5))

    conv3 = nn.Conv2d(in_channels=64,
                      out_channels=256,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False)
    init.normal_(conv3.weight, mean=0.0, std=0.02)
    net3 = nn.Sequential(conv3, nn.LeakyReLU(negative_slope=0.2), nn.Dropout(0.5))

    # split
    conv4 = nn.Conv2d(in_channels=64,
                      out_channels=1,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False)
    init.normal_(conv4.weight, mean=0.0, std=0.02)
    net4 = nn.Sequential(conv4)

    # whole again
    conv5 = nn.Conv2d(in_channels=4,
                      out_channels=1,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False)
    init.normal_(conv5.weight, mean=0.0, std=0.02)
    net5 = nn.Sequential(conv5)

    x = torch.ones(1, 1, 64, 84)
    print(x.shape)

    x = net1(x)
    print(x.shape)

    x = net2(x)
    print(x.shape)

    x = net3(x)
    print(x.shape)

    x1, x2, x3, x4 = x.split([64, 64, 64, 64], dim=1)
    print(x1.shape)

    x1 = net4(x1)
    x2 = net4(x2)
    x3 = net4(x3)
    x4 = net4(x4)

    x = torch.cat([x1, x2, x3, x4], dim=1)
    print(x.shape)

    x = net5(x)
    print(x.shape)


    # x = net3(x)
    # print(x.shape)

    # x = net4(x)
    # print(x.shape)


if __name__ == '__main__':
    test_deeper_d()
