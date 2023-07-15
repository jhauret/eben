"""
PseudoQMFBanks class

PQMF formalism is introduced in:
"Nguyen, T. Q. (1994). Near-perfect-reconstruction pseudo-QMF banks.
 IEEE Transactions on signal processing, 42(1), 65-76."

The design methodology is introduced in:
"Lin, Y. P., & Vaidyanathan, P. P. (1998). A Kaiser window approach for the design of
prototype filters of cosine modulated filterbanks. IEEE signal processing letters, 5(6), 132-134."
"""

import torch
from torch import pi, nn


class PseudoQMFBanks(nn.Module):
    """
    PQMF class used to compute:
        * analysis from temporal signal
        * synthesis from decomposed representation

    The PQMF weights are initialized using the `initialize_pqmf_bank` method.
    """
    def __init__(self, decimation: int = 32, kernel_size: int = 1024, beta: int = 9):
        """
        Initialize the PseudoQMFBanks module.

        Args:
            decimation (int): The decimation factor, noted "M" in the article.
            kernel_size (int): The length of the PQMF kernel. Convolutions with longer kernels are slower to compute but
                                allow for a lower reconstruction error and better band separation. Minimal band overlap
                                implies a very low reconstruction error, but it is not equivalent. Indeed, phase
                                opposition phenomena between the bands also contribute to eliminating redundant content
                                in the synthesis phase. In practice, a kernel_size of 8 * decimation is sufficient for
                                 pseudo-perfect reconstruction, and a kernel_size of 128 * decimation is sufficient for
                                 pseudo-perfect separation of frequency content between bands.
            beta (int): The beta value used in the Kaiser window function.
        """
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
        """
        Compute the PQMF prototype filter.

        Args:
            cutoff_ratio (float): The cutoff ratio of the Kaiser window.

        Returns:
            torch.Tensor: The computed prototype filter.
        """

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
        """
        Compute the optimal cutoff ratio used in the Kaiser window to make the prototype filter:
         - close to zero out of its passband to minimize aliasing
         - close to one within its passband to minimize distortion

        Returns:
            float: The optimal cutoff ratio.
        """
        def objective(cutoff):
            """
            Compute Equation (5) of the Y.Lin article. The lower, the better.
            """
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

        # Perform 5 optimization steps to find the optimal cutoff_ratio
        for _ in range(5):
            optimizer.zero_grad()
            criterion = objective(cutoff_ratio_lbfgs)
            criterion.backward()
            optimizer.step(lambda: objective(cutoff_ratio_lbfgs))

        return cutoff_ratio_lbfgs.item()

    def initialize_pqmf_bank(self):
        """
        Initialize the PQMF analysis and synthesis weights according to Equation (1) of the T.Q Nguyen article.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Analysis and synthesis weights.
        """
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
        """
        Forward pass of the PQMF module.

        Args:
            signal (torch.Tensor): The input signal.
            stage (str): The stage of processing, either "analysis" or "synthesis".
            bands (str or int): The number of bands to compute starting from first ones.
                                'all' for all bands, or an integer value.

        Returns:
            torch.Tensor: The output signal after analysis or synthesis.
        """

        if stage == "analysis":
            if bands == 'all':
                # Compute all bands
                return torch.nn.functional.conv1d(signal, self.analysis_weights, bias=None,
                                                  stride=(self._decimation,),
                                                  padding=(self._kernel_size - 1,))
            else:
                # Compute only the first bands
                return torch.nn.functional.conv1d(signal, self.analysis_weights[:bands, :, :],
                                                  bias=None, stride=(self._decimation,),
                                                  padding=(self._kernel_size - 1,))
        elif stage == "synthesis":
            # Number of channels is equal to the decimation factor
            return torch.nn.functional.conv_transpose1d(signal, self.synthesis_weights, bias=None,
                                                        stride=(self._decimation,),
                                                        output_padding=self._decimation - 2,
                                                        groups=self._decimation,
                                                        padding=(self._kernel_size - 1,))
        else:
            raise ValueError(f"Stage: {stage} is not recognized.")

    def cut_tensor(self, tensor):
        """
        Cut the tensor's dimension 2 length to be divisible by _decimation.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with dimension 2 length divisible by _decimation.
        """

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
