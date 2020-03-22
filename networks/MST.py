import torch.nn as nn
import torch
from torch.nn import init
from networks.util import ResnetBlock


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        leaky = nn.LeakyReLU(negative_slope=0.2)

        # shape = (64, 84, 1)
        # df_dim = 64

        conv1 = nn.Conv2d(in_channels=1,
                           out_channels=64,
                           kernel_size=7,
                           stride=2,
                           padding=3,
                           bias=False)
        init.normal_(conv1.weight, mean=0.0, std=0.02)
        self.net1 = nn.Sequential(conv1, leaky)

        conv2 = nn.Conv2d(in_channels=64,
                               out_channels=256,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        init.normal_(conv2.weight, mean=0.0, std=0.02)
        instance_norm = nn.InstanceNorm2d(256, eps=1e-5)
        self.net2 = nn.Sequential(conv2, instance_norm, leaky)

        conv3 = nn.Conv2d(in_channels=256,
                               out_channels=1,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        init.normal_(conv3.weight, mean=0.0, std=0.02)
        self.net3 = nn.Sequential(conv3, leaky)

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)
        x = self.net1(x)
        # (batch * 64 * 32 * 42)
        x = self.net2(x)
        # (batch * 256 * 16 * 21)
        x = self.net3(x)
        # (batch * 1 * 16 * 21)

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        padding = nn.ReflectionPad2d((3, 3, 3, 3))

        relu = nn.ReLU()

        conv1 = nn.Conv2d(in_channels=1,
                          out_channels=64,
                          kernel_size=7,
                          stride=1,
                          padding=0,
                          bias=False)
        init.normal_(conv1.weight)
        instance_norm = nn.InstanceNorm2d(64, eps=1e-5)
        self.cnet1 = nn.Sequential(padding, conv1, instance_norm, relu)

        conv2 = nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False)
        init.normal_(conv2.weight)
        instance_norm = nn.InstanceNorm2d(128, eps=1e-5)
        self.cnet2 = nn.Sequential(conv2, instance_norm, relu)

        conv3 = nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=False)
        init.normal_(conv3.weight)
        instance_norm = nn.InstanceNorm2d(256, eps=1e-5)
        self.cnet3 = nn.Sequential(conv3, instance_norm, relu)

        self.resnet = nn.Sequential()
        for i in range(10):
            self.resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                          padding_type='reflect',
                                                          use_dropout=False,
                                                          use_bias=False,
                                                          norm_layer=nn.InstanceNorm2d))
        extra_padding = nn.ZeroPad2d((0, 1, 0, 1))

        tconv1 = nn.ConvTranspose2d(in_channels=256,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False)

        init.normal_(tconv1.weight)
        self.tcnet1 = nn.Sequential(tconv1, extra_padding, instance_norm, relu)

        tconv2 = nn.ConvTranspose2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False)
        init.normal_(tconv2.weight)
        self.tcnet2 = nn.Sequential(tconv2, extra_padding, instance_norm, relu)

        conv4 = nn.Conv2d(in_channels=64,
                          out_channels=1,
                          kernel_size=7,
                          stride=1,
                          padding=0)
        init.normal_(conv4.weight)
        sigmoid = nn.Sigmoid()

        self.cnet4 = nn.Sequential(padding, conv4, sigmoid)

    def forward(self, tensor_in):
        x = tensor_in
        # (batch * 1 * 64 * 84)

        x = self.cnet1(x)
        # (batch * 1 * 70 * 90) after padded
        # (batch * 64 * 64 * 84)
        x = self.cnet2(x)
        # (batch * 128 * 32 * 42)
        x = self.cnet3(x)
        # (batch * 256 * 16 * 21)

        x = self.resnet(x)
        # (batch * 256 * 16 * 21)

        x = self.tcnet1(x)
        # (batch * 128 * 32 * 42)
        x = self.tcnet2(x)
        # (batch * 64 * 64 * 84)
        x = self.cnet4(x)
        # After padding, (batch * 64 * 70 * 90)
        # (batch * 1 * 64 * 84)

        return x

def test_g():
    padding = nn.ReflectionPad2d((3, 3, 3, 3))

    relu = nn.ReLU()

    conv1 = nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=7,
                      stride=1,
                      padding=0,
                      bias=False)
    init.normal_(conv1.weight)
    instance_norm = nn.InstanceNorm2d(64, eps=1e-5)
    cnet1 = nn.Sequential(padding, conv1, instance_norm, relu)

    conv2 = nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False)
    init.normal_(conv2.weight)
    instance_norm = nn.InstanceNorm2d(128, eps=1e-5)
    cnet2 = nn.Sequential(conv2, instance_norm, relu)

    conv3 = nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False)
    init.normal_(conv3.weight)
    instance_norm = nn.InstanceNorm2d(256, eps=1e-5)
    cnet3 = nn.Sequential(conv3, instance_norm, relu)

    resnet = nn.Sequential()
    for i in range(10):
        resnet.add_module('resnet_block', ResnetBlock(dim=256,
                                                           padding_type='reflect',
                                                           use_dropout=False,
                                                           use_bias=False,
                                                           norm_layer=nn.InstanceNorm2d))
    extra_padding = nn.ZeroPad2d((0, 1, 0, 1))

    tconv1 = nn.ConvTranspose2d(in_channels=256,
                                out_channels=128,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False)

    init.normal_(tconv1.weight)
    tcnet1 = nn.Sequential(tconv1, extra_padding, instance_norm, relu)

    tconv2 = nn.ConvTranspose2d(in_channels=128,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False)
    init.normal_(tconv2.weight)
    tcnet2 = nn.Sequential(tconv2, extra_padding, instance_norm, relu)

    conv4 = nn.Conv2d(in_channels=64,
                      out_channels=1,
                      kernel_size=7,
                      stride=1,
                      padding=0)
    init.normal_(conv4.weight)
    sigmoid = nn.Sigmoid()

    cnet4 = nn.Sequential(padding, conv4, sigmoid)


    x = torch.ones((4, 1, 64, 84))
    x = cnet1(x)
    x = cnet2(x)
    x = cnet3(x)
    x = resnet(x)
    x = tcnet1(x)
    x = tcnet2(x)
    x = cnet4(x)
    print(x.shape)

if __name__ == '__main__':
    test_g()
