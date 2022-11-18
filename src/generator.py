
""" EBEN generator and sub blocks definition in Pytorch"""

import torch
from torch import nn
from src.pqmf import PseudoQMFBanks


class GeneratorEBEN(nn.Module):

    def __init__(self, bands_nbr: int, pqmf_ks: int):
        """
        Generator of EBEN

        bands_nbr: number of pqmf bands
        pqmf_ks: kernel size of pqmf
        """
        super().__init__()

        self.pqmf_ks = pqmf_ks

        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.pqmf = PseudoQMFBanks(decimation=bands_nbr, kernel_size=pqmf_ks)

        self.multiple = 2 * 4 * 8 * 4  # strides=2*4*8 * bands_nbr=4

        self.first_conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3,
                                    padding='same', bias=False, padding_mode='reflect')

        self.encoder_blocks = nn.ModuleList(
            [EncBlock(out_channels=64, stride=2, nl=self.nl),
             EncBlock(out_channels=128, stride=4, nl=self.nl),
             EncBlock(out_channels=256, stride=8, nl=self.nl)])

        self.latent_conv = nn.Sequential(self.nl,
                                         normalized_conv1d(in_channels=256, out_channels=64,
                                                           kernel_size=7, padding='same',
                                                           bias=False, padding_mode='reflect'),
                                         self.nl,
                                         normalized_conv1d(in_channels=64, out_channels=256,
                                                           kernel_size=7, padding='same',
                                                           bias=False, padding_mode='reflect'),
                                         self.nl)

        self.decoder_blocks = nn.ModuleList(
            [DecBlock(out_channels=128, stride=8, nl=self.nl),
             DecBlock(out_channels=64, stride=4, nl=self.nl),
             DecBlock(out_channels=32, stride=2, nl=self.nl)])

        self.last_conv = nn.Conv1d(in_channels=32, out_channels=4, kernel_size=3,
                                   padding='same', bias=False, padding_mode='reflect')

    def forward(self, cut_audio):
        """
        Forward pass of generator.
        Args:
            cut_audio: in-ear speech signal
        """

        # PQMF analysis, for first band only
        first_band = self.pqmf(cut_audio, "analysis", bands=1)

        # First conv
        x = self.first_conv(first_band)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))

        # Latent forward
        x = self.latent_conv(x3)

        # Decoder forward
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        # Recompose PQMF bands ( + avoiding any inplace operation for backprop )
        b, c, t = first_band.shape  # (batch_size, channels, time_len)
        fill_up_tensor = torch.zeros((b, 3, t), requires_grad=False).type_as(first_band)
        cat_tensor = torch.cat(tensors=(first_band, fill_up_tensor), dim=1)
        speech_decomposed_processed = torch.tanh(x + cat_tensor)
        enhanced_speech = torch.sum(self.pqmf(speech_decomposed_processed, "synthesis"), 1,
                                    keepdim=True)

        return enhanced_speech, speech_decomposed_processed

    def cut_tensor(self, tensor):
        """ This function is used to make tensor's dim 2 len divisible by multiple """

        old_len = tensor.shape[2]
        new_len = old_len - (old_len + self.pqmf_ks) % self.multiple
        tensor = torch.narrow(tensor, 2, 0, new_len)

        return tensor


class DecBlock(nn.Module):
    """
    Decoder Block Module
    """
    def __init__(self, out_channels, stride, nl, bias=False):
        super().__init__()

        self.nl = nl

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels, nl=nl, dilation=1),
            ResidualUnit(channels=out_channels, nl=nl, dilation=3),
            ResidualUnit(channels=out_channels, nl=nl, dilation=9))

        self.conv_trans = normalized_conv_trans1d(in_channels=2 * out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=2 * stride, stride=stride,
                                                  padding=stride // 2,
                                                  output_padding=0, bias=bias)

    def forward(self, x, encoder_output):
        x = x + encoder_output
        out = self.residuals(self.nl(self.conv_trans(x)))
        return out


class EncBlock(nn.Module):
    """
    Encoder Block Module
    """
    def __init__(self, out_channels, stride, nl, bias=False):
        super().__init__()

        self.nl = nl

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=1),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=3),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=9))
        self.conv = normalized_conv1d(in_channels=out_channels // 2, out_channels=out_channels,
                                      kernel_size=2 * stride,
                                      stride=stride, padding=stride - 1, bias=bias,
                                      padding_mode='reflect')

    def forward(self, x):
        out = self.conv(self.residuals(x))
        return out


class ResidualUnit(nn.Module):
    """
    Residual Unit Module
    """
    def __init__(self, channels, nl, dilation, bias=False):
        super().__init__()

        self.dilated_conv = normalized_conv1d(in_channels=channels, out_channels=channels,
                                                    kernel_size=3,
                                                    dilation=dilation, padding='same', bias=bias,
                                                    padding_mode='reflect')
        self.pointwise_conv = normalized_conv1d(in_channels=channels, out_channels=channels,
                                                kernel_size=1,
                                                padding='same', bias=bias, padding_mode='reflect')
        self.nl = nl

    def forward(self, x):
        out = x + self.nl(self.pointwise_conv(self.dilated_conv(x)))
        return out


def normalized_conv1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


def normalized_conv_trans1d(*args, **kwargs):
    return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


if __name__ == '__main__':

    # Instantiate nn.module
    generator = GeneratorEBEN(bands_nbr=4, pqmf_ks=32)

    # Instantiate tensor with shape: (batch_size, channel, time_len)
    corrupted_signal = torch.randn((5, 1, 60000))
    #  cut tensor to ensure forward pass run well
    corrupted_signal = generator.cut_tensor(corrupted_signal)

    # Test forward of model
    enhanced_signal, enhanced_signal_decomposed= generator(corrupted_signal)

    # Number of parameters
    pytorch_total_params = sum(p.numel() for p in generator.parameters())
    print(f"pytorch_total_params: {pytorch_total_params*1e-6:.2f} Millions")
