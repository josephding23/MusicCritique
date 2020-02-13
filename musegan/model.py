import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


class MuseGenerator(nn.Module):
    def __init__(self, n_tracks, beat_resolution):
        super(MuseGenerator, self).__init__()
        self.n_tracks = n_tracks
        self.beat_resolution = beat_resolution

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
    def __init__(self, n_tracks):
        super(MuseDiscriminator, self).__init__()
        self.n_tracks = n_tracks

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