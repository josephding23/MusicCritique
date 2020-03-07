import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
from torch.optim import lr_scheduler
import torch.nn.functional as F


class MuseGenerator(nn.Module):
    def __init__(self, opt):
        super(MuseGenerator, self).__init__()
        self.n_tracks = opt.n_tracks
        self.beat_resolution = opt.beat_resolution

        self.shared = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=512, kernel_size=(4, 1, 1), stride=(4, 1, 1)), # 4, 1, 1
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=256, kernel_size=(1, 4, 3), stride=(1, 4, 3)), # 4, 4, 3
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=128, kernel_size=(1, 4, 3), stride=(1, 4, 2)), # 4, 16, 7
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        self.pitch_time_private_1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=32, kernel_size=(1, 1, 12), stride=(1, 1, 12)), # 4, 16, 84
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.pitch_time_private_2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=16, kernel_size=(1, 3, 1), stride=(1, 3, 1)), # 4, 48, 84
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.time_pitch_private_1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=32, kernel_size=(1, 3, 1), stride=(1, 3, 1)), # 4, 48, 7
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.time_pitch_private_2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=16, kernel_size=(1, 1, 12), stride=(1, 1, 12)), # 4, 48, 84
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.merged_private =nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.n_tracks, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1)), # 4, 48, 84
            nn.BatchNorm3d(32)
        )

    def forward(self, tensor_in):
        h = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(tensor_in, 1), 1), 1)
        h = self.shared(h)

        s1 = [ self.pitch_time_private_1(h) for _ in range(self.n_tracks) ]
        s1 = [ self.pitch_time_private_2(s1[i]) for i in range(self.n_tracks)]

        s2 = [ self.time_pitch_private_1(h) for _ in range(self.n_tracks) ]
        s2 = [ self.time_pitch_private_2(s2[i]) for i in range(self.n_tracks)]

        h = [ torch.cat((s1[i], s2[i]), -1) for i in range(self.n_tracks) ]
        h = [ self.merged_private(h[i]) for i in range(self.n_tracks) ]
        h = torch.cat(h, -1)

        return torch.tanh(h)


class MuseDiscriminator(nn.Module):
    def __init__(self, opt):
        super(MuseDiscriminator, self).__init__()
        self.n_tracks = opt.n_tracks

        self.pitch_time_private_1 = nn.Sequential(
            nn.Conv3d(self.n_tracks, 16, (1, 1, 12), (1, 1, 12)),
            nn.LeakyReLU()
        )

        self.pitch_time_private_2 = nn.Sequential(
            nn.Conv3d(self.n_tracks, 32, (1, 3, 1), (1, 3, 1)),
            nn.LeakyReLU()
        )

        self.time_pitch_private_1 = nn.Sequential(
            nn.Conv3d(self.n_tracks, 16, (1, 3, 1), (1, 3, 1)),
            nn.LeakyReLU()
        )

        self.time_pitch_private_2 = nn.Sequential(
            nn.Conv3d(self.n_tracks, 32, (1, 1, 12), (1, 1, 12)),
            nn.LeakyReLU()
        )

        self.merged = nn.Sequential(
            nn.Conv3d(self.n_tracks, 64, (1, 1, 1), (1, 1, 1)),
            nn.LeakyReLU()
        )

        self.shared = nn.Sequential(
            nn.Conv3d(self.n_tracksm, 128, (1, 4, 3), (1, 4, 3)),
            nn.LeakyReLU(),
            nn.Conv3d(self.n_tracks, 256, (1, 4, 3), (1, 4, 3)),
            nn.LeakyReLU()
        )

        self.chroma = nn.Sequential(
            nn.Conv3d(self.n_tracks, 16, (1, 3, 1), (1, 3, 1)),
            nn.LeakyReLU(),
            nn.Conv3d(self.n_tracks, 32, (1, 4, 1), (1, 4, 1)),
            nn.LeakyReLU(),
            nn.Conv3d(self.n_tracks, 64, (1, 4, 1), (1, 4, 1)),
            nn.LeakyReLU()
        )

        self.on_off_set = nn.Sequential(
            nn.Conv3d(self.n_tracks, 16, (1, 3, 1), (1, 3, 1)),
            nn.LeakyReLU(),
            nn.Conv3d(self.n_tracks, 32, (1, 4, 1), (1, 4, 1)),
            nn.LeakyReLU(),
            nn.Conv3d(self.n_tracks, 64, (1, 4, 1), (1, 4, 1)),
            nn.LeakyReLU()
        )

        self.dense = nn.Linear(self.n_tracks, 1)

    def forward(self, tensor_in):
        h = tensor_in
        n_beats = h.shape[2]
        reshaped = torch.reshape(
            tensor_in, (-1, h.shape[1], n_beats, self.beat_resolution,
                            h.shape[3], h.shape[4]))
        summed = torch.sum(reshaped, 3)

        factor = int(h.shape[3]) // 12
        remainder = int(h.shape[3]) % 12
        reshaped = torch.reshape(
            summed[..., (factor * 12), :],
            (-1, h.shape[1], n_beats, factor, 12, h.shape[4]))

        chroma = torch.sum(reshaped, 3)
        if remainder:
            chroma += summed[..., -remainder:, :]

        padded = F.pad(tensor_in[:, :, :-1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

        on_off_set = torch.sum(tensor_in - padded, 3, True)

        s1 = [self.pitch_time_private_1(h) for _ in range(self.n_tracks)]
        s1 = [self.pitch_time_private_2(s1[i]) for i in range(self.n_tracks)]

        s2 = [self.time_pitch_private_1(h) for _ in range(self.n_tracks)]
        s2 = [self.time_pitch_private_2(s2[i]) for i in range(self.n_tracks)]

        h = [torch.cat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

        h = [self.merged_private(h[i]) for i in range(self.n_tracks)]

        h = torch.cat(h, -1)

        h = self.shared(h)
        c = self.chroma(h)

        o = self.on_off_set(on_off_set)

        h = torch.cat((h, c, o), -1)
        h = self.merged(h)
        h = torch.reshape(h, (-1, h.shape[-1]))
        h = self.dense(h)

        return h


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        # return target_tensor.expand_as(prediction)
        return target_tensor.expand_as(prediction).to(torch.device('cuda'))

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            # prediction is 2D prediction map vector from Discriminator
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # loss = self.loss(prediction, target_tensor)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def test():
    zeros = np.zeros((4, 120, 84, 5), dtype=np.bool_)