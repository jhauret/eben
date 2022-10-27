
""" EBEN discriminator and sub blocks definition in Pytorch"""

import torch
from torch import nn


class DiscriminatorEBENMultiScales(nn.Module):
    """
    EBEN overall Multiscales Discriminator: 3 bands discriminators + 1 melgan
    """
    def __init__(self):
        super().__init__()

        # PQMF discriminators
        self.pqmf_discriminators = torch.nn.ModuleList()
        for dila in [1, 2, 3]:
            self.pqmf_discriminators.append(DiscriminatorEBEN(dilation=dila))

        # MelGAN discriminator
        self.melgan_discriminator = DiscriminatorMelGAN()

    def forward(self, bands, audio):

        embeddings = []

        for dis in self.pqmf_discriminators:
            embeddings.append(dis(bands))

        embeddings.append(self.melgan_discriminator(audio))

        return embeddings


class DiscriminatorEBEN(nn.Module):
    """
    EBEN PQMF-bands discriminator
    """
    def __init__(self, dilation=1):
        super().__init__()

        self.dilation = dilation

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(1),
                normalized_conv1d(3, 30, kernel_size=(3,), stride=(1,), padding=(1,),
                                  dilation=self.dilation, groups=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(30, 60, kernel_size=(7,), stride=(2,), padding=(3,),
                                  dilation=self.dilation, groups=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(60, 120, kernel_size=(7,), stride=(2,), padding=(3,),
                                  dilation=self.dilation, groups=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(120, 240, kernel_size=(7,), stride=(2,), padding=(3,),
                                  dilation=self.dilation, groups=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(240, 480, kernel_size=(7,), stride=(2,), padding=(3,),
                                  dilation=self.dilation, groups=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(480, 960, kernel_size=(7,), stride=(2,), padding=(3,),
                                  dilation=self.dilation, groups=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(960, 960, kernel_size=(5,), stride=(1,), padding=(2,),
                                  dilation=self.dilation, groups=3),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            normalized_conv1d(960, 1, kernel_size=(3,), stride=(1,), padding=(1,), groups=1),
        ])

    def forward(self, bands):
        embeddings = [bands]
        for module in self.discriminator:
            embeddings.append(module(embeddings[-1]))
        return embeddings


class DiscriminatorMelGAN(nn.Module):
    """
    MelGAN Discriminator
     inspired from https://github.com/seungwonpark/melgan/blob/master/model/discriminator.py
    """
    def __init__(self):
        super().__init__()


        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                normalized_conv1d(in_channels=1, out_channels=16, kernel_size=(15,), stride=(1,)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(in_channels=16, out_channels=64, kernel_size=(41,), stride=(4,),
                                  padding=20, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(in_channels=64, out_channels=256, kernel_size=(41,), stride=(4,),
                                  padding=20, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(in_channels=256, out_channels=1024, kernel_size=(41,),
                                  stride=(4,), padding=20, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(in_channels=1024, out_channels=1024, kernel_size=(41,),
                                  stride=(4,), padding=20, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                normalized_conv1d(in_channels=1024, out_channels=1024, kernel_size=(5,),
                                  stride=(1,), padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            normalized_conv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, audio):
        embeddings = [audio]
        for module in self.discriminator:
            embeddings.append(module(embeddings[-1]))
        return embeddings


def normalized_conv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


if __name__ == '__main__':

    # Instantiate nn.modules
    dis_bands = DiscriminatorEBEN()
    dis_melgan = DiscriminatorMelGAN()
    dis_ms = DiscriminatorEBENMultiScales()

    # Instantiate tensors with shape: (batch_size, channel, time_len)
    pqmf_bands = torch.randn((5, 3, 1500))
    speech = torch.randn((5, 1, 60000))

    # Test forward of models
    scores_bands = dis_bands(bands=pqmf_bands)
    scores_melgan = dis_melgan(audio=speech)
    scores_ms = dis_ms(bands=pqmf_bands, audio=speech)

    # Number of parameters
    bands_params = sum(p.numel() for p in dis_bands.parameters())
    melgan_params = sum(p.numel() for p in dis_melgan.parameters())
    ms_params = sum(p.numel() for p in dis_ms.parameters())
    print(f"DiscriminatorEBEN has {bands_params * 1e-6:.2f} M parameters")
    print(f"DiscriminatorMelGAN has {melgan_params*1e-6:.2f} M parameters")
    print(f"DiscriminatorEBENMultiScales has {ms_params*1e-6:.2f} M parameters")
