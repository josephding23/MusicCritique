import torch.nn as nn
import torch
from torch.nn import init


def init_weight_(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.02)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # shape = (64, 84, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=[1, 12],
                      stride=[1, 12],
                      padding=0,
                      bias=False
                      ),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=[4, 1],
                      stride=[4, 1],
                      padding=0,
                      bias=False
                      ),
            nn.InstanceNorm2d(128, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=[2, 1],
                      stride=[2, 1],
                      padding=0,
                      bias=False
                      ),
            nn.InstanceNorm2d(256, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=[8, 1],
                      stride=[8, 1],
                      padding=0,
                      bias=False
                      ),
            nn.InstanceNorm2d(512, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=512,
                      out_channels=2,
                      kernel_size=[1, 7],
                      stride=[1, 7],
                      padding=0,
                      bias=False
                      ),
            # nn.Sigmoid()
        )

        init_weight_(self.net)

    def forward(self, tensor_in):
        x = tensor_in
        x = self.net(x)
        x = x.view(-1, 2)
        x = torch.sigmoid(x)
        return x


def test_classifier():
    conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1,
                  out_channels=64,
                  kernel_size=[1, 12],
                  stride=[1, 12],
                  padding=0,
                  bias=False
                  ),
        nn.LeakyReLU(negative_slope=0.2),
    )

    conv2 = nn.Sequential(
        nn.Conv2d(in_channels=64,
                  out_channels=128,
                  kernel_size=[4, 1],
                  stride=[4, 1],
                  padding=0,
                  bias=False
                  ),
        nn.InstanceNorm2d(128, eps=1e-5),
        nn.LeakyReLU(negative_slope=0.2),
    )

    conv3 = nn.Sequential(
        nn.Conv2d(in_channels=128,
                  out_channels=256,
                  kernel_size=[2, 1],
                  stride=[2, 1],
                  padding=0,
                  bias=False
                  ),
        nn.InstanceNorm2d(256, eps=1e-5),
        nn.LeakyReLU(negative_slope=0.2),
    )

    conv4 = nn.Sequential(
        nn.Conv2d(in_channels=256,
                  out_channels=512,
                  kernel_size=[8, 1],
                  stride=[8, 1],
                  padding=0,
                  bias=False
                  ),
        nn.InstanceNorm2d(512, eps=1e-5),
        nn.LeakyReLU(negative_slope=0.2),
    )

    conv5 = nn.Sequential(
        nn.Conv2d(in_channels=512,
                  out_channels=2,
                  kernel_size=[1, 7],
                  stride=[1, 7],
                  padding=0,
                  bias=False
                  )
    )

    soft = nn.Softmax(dim=1)

    x = torch.zeros((16, 1, 64, 84))
    print(x.shape)

    x = conv1(x)
    print(x.shape)

    x = conv2(x)
    print(x.shape)

    x = conv3(x)
    print(x.shape)

    x = conv4(x)
    print(x.shape)

    x = conv5(x)
    print(x.shape)

    x = soft(x)
    print(x.shape)

    x = x.view(-1, 2)
    print(x.shape)


if __name__ == '__main__':
    test_classifier()
