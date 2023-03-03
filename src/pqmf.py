
""" PseudoQMFBanks class"""

import torch
from torch import pi, nn


class PseudoQMFBanks(nn.Module):
    """
    PQMF class used to compute:
        * analysis from temporal signal
        * synthesis from decomposed representation

    PQMF weights are initialized thanks to initialize_pqmf_bank
    """
    def __init__(self, decimation: int = 32, kernel_size: int = 1024, beta: int = 9):
        super().__init__()
        assert kernel_size % (4 * decimation) == 0

        self._decimation = decimation
        self._kernel_size = kernel_size
        self._beta = beta

        # real initializations are found below
        self._cutoff_ratio = self.initialize_cutoff_ratio()
        analysis_weights, synthesis_weights = self.initialize_pqmf_bank()
        self.analysis_weights = nn.parameter.Parameter(data=analysis_weights, requires_grad=False)
        self.synthesis_weights = nn.parameter.Parameter(data=synthesis_weights, requires_grad=False)

    def compute_prototype(self, cutoff_ratio):
        kaiser = torch.ones(1, 1, self._kernel_size, dtype=torch.double)
        kaiser[0, 0, :] = torch.kaiser_window(self._kernel_size, periodic=False, beta=self._beta,
                                              requires_grad=True)

        sinc = torch.ones(1, 1, self._kernel_size)
        sinc[0, 0, :] = cutoff_ratio * torch.special.sinc(
            cutoff_ratio * (torch.arange(self._kernel_size) - (self._kernel_size - 1) / 2))

        prototype = torch.ones(1, 1, self._kernel_size)
        prototype[0, 0, :] = sinc * kaiser

        return prototype

    def initialize_cutoff_ratio(self):
        def objective(cutoff):
            prototype = self.compute_prototype(cutoff)
            proto_padded = nn.functional.pad(prototype,
                                             pad=(self._kernel_size // 2, self._kernel_size // 2),
                                             mode='constant', value=0.0)
            autocorr_proto = nn.functional.conv1d(proto_padded, prototype)
            autocorr_proto[..., self._kernel_size // 2] = 0
            autocorr_proto_2m = autocorr_proto[..., ::2 * self._decimation]
            phi_new = torch.max(torch.abs(autocorr_proto_2m))

            if abs(cutoff - 1 / (2 * self._decimation)) > 1 / (4 * self._decimation):
                penalty = 1 / (4 * self._decimation)
            else:
                penalty = 0

            return phi_new + penalty

        cutoff_ratio_lbfgs = torch.ones(1) / (2 * self._decimation)
        cutoff_ratio_lbfgs.requires_grad = True

        optimizer = torch.optim.LBFGS([cutoff_ratio_lbfgs], line_search_fn="strong_wolfe")

        for _ in range(5):
            optimizer.zero_grad()
            criterion = objective(cutoff_ratio_lbfgs)
            criterion.backward()
            optimizer.step(lambda: objective(cutoff_ratio_lbfgs))

        return cutoff_ratio_lbfgs.item()

    def initialize_pqmf_bank(self):
        prototype = self.compute_prototype(self._cutoff_ratio).squeeze()
        analysis_weights = torch.zeros(self._decimation, 1, self._kernel_size)
        synthesis_weights = torch.zeros(self._decimation, 1, self._kernel_size)
        for pqmf_idx in range(self._decimation):
            analysis_weights[pqmf_idx, 0, :] = 2 * torch.flip(prototype * torch.cos(
                (2 * pqmf_idx + 1) * pi / 2 / self._decimation * (
                        torch.arange(self._kernel_size) - (self._kernel_size - 1) / 2) + (
                    -1) ** pqmf_idx * pi / 4), [0])

            synthesis_weights[pqmf_idx, 0, :] = self._decimation * 2 * prototype * torch.cos(
                (2 * pqmf_idx + 1) * pi / 2 / self._decimation * (
                        torch.arange(self._kernel_size) - (self._kernel_size - 1) / 2) - (
                    -1) ** pqmf_idx * pi / 4)

        return analysis_weights, synthesis_weights

    def forward(self, signal, stage, bands='all'):
        if stage == "analysis":
            if bands == 'all':
                # compute all bands
                return torch.nn.functional.conv1d(signal, self.analysis_weights, bias=None,
                                                  stride=(self._decimation,),
                                                  padding=(self._kernel_size - 1,))
            else:
                # compute only the first bands
                return torch.nn.functional.conv1d(signal, self.analysis_weights[:bands, :, :],
                                                  bias=None, stride=(self._decimation,),
                                                  padding=(self._kernel_size - 1,))
        elif stage == "synthesis":
            # number of channels is equal to decimation factor
            return torch.nn.functional.conv_transpose1d(signal, self.synthesis_weights, bias=None,
                                                        stride=(self._decimation,),
                                                        output_padding=self._decimation - 2,
                                                        groups=self._decimation,
                                                        padding=(self._kernel_size - 1,))
        else:
            raise ValueError(f"stage: {stage} is not recognized by {self._get_name()}")

    def cut_tensor(self, tensor):
        """ This function is used to make tensor's dim 2 len divisible by _decimation """

        old_len = tensor.shape[2]
        new_len = old_len - (old_len + self._kernel_size) % self._decimation
        tensor = torch.narrow(tensor, 2, 0, new_len)

        return tensor


if __name__ == '__main__':

    # Instantiate nn.module
    pqmf = PseudoQMFBanks(decimation=4, kernel_size=32)

    # Instantiate tensors with shape: (batch_size, channel, time_len)
    audio = pqmf.cut_tensor(torch.rand(4, 1, 48009))

    # Test analysis and synthesis
    audio_decomposed = pqmf(audio, "analysis")
    audio_recomposed = torch.sum(pqmf(audio_decomposed, "synthesis"), 1, keepdim=True)

    # Statistics
    print(f'Original signal length: {audio.shape[2]} with {audio.shape[1]} channel')
    print(f'Decomposed signal length: {audio_decomposed.shape[2]} with {audio_decomposed.shape[1]} channels')
    print(f'Recomposed signal length: {audio_recomposed.shape[2]} with {audio_recomposed.shape[1]} channel')
    snr = 10 * torch.log10((audio_recomposed ** 2).mean() / ((audio - audio_recomposed) ** 2).mean()).item()
    print(f'SNR of chirp_recomposed: {snr:.2f}dB')
    pqmf_params = sum(p.numel() for p in pqmf.parameters())
    print(f"PQMF params: {pqmf_params}")
