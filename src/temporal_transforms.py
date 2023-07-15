""" Definition of several audio transformation methods in the TemporalTransforms class  """

import torch
from torchaudio.functional import lowpass_biquad
from torchaudio.transforms import Resample


class TemporalTransforms:
    """
    Processing class to apply transforms on an audio signal.

    Args:
        audio: torch.Tensor,
            audio waveform to process

        sr: int,
            sampling rate of the audio

        padding_length: int,
            used for IRR response stabilisation

        deterministic: bool,
            whether to apply a deterministic filter and audio selection

    """

    def __init__(
        self,
        audio: torch.Tensor,
        sr: int,
        padding_length: int = 10000,
        deterministic: bool = False,
    ):
        self._audio = audio
        self._sr = sr
        self.padding_length = padding_length
        self.pad = torch.nn.ReflectionPad1d(self.padding_length)
        self.determinist = deterministic

    @property
    def audio(self):
        """
        Get audio signal

        Returns:
            self._audio:torch.Tensor,
                audio of the class, with shape [1,time_len]
        """

        return self._audio

    @property
    def sr(self):
        """
        Get sampling rate of the signal

        Returns:
            self._sr:int,
        """

        return self._sr

    def smoothing(self):
        """
        Smooth the signal's borders to get rid of any jump
        """

        smoothing_len = 512
        self._audio[0, :smoothing_len] *= torch.pow(
            torch.linspace(0, 1, smoothing_len), 2
        )
        self._audio[0, -smoothing_len:] *= torch.pow(
            torch.linspace(1, 0, smoothing_len), 2
        )

    def add_noise(self, intensity: float = 0.005):
        """
        Add gaussian noise to the signal

            Args:
                intensity: float,
                    intensity of the noise
        """

        mean = torch.zeros(size=self._audio.shape)
        std = intensity * self._audio.std() * torch.ones(size=self._audio.shape)
        self._audio += torch.normal(mean=mean, std=std)

    def remove_hf(self, cutoff_freq: int = 600, q_factor: float = 1):
        """
        Low-pass filter of the fourth order

            Args:

                cutoff_freq:int,
                    cutoff frequency of the filter

                Q:float,
                    Quality factor of the filter
        """
        # pad for IRR response stabilisation
        self._audio = self.pad(self._audio)

        # filt-filt trick for 0-phase shift
        if not self.determinist:
            rand_factors = torch.FloatTensor(2).uniform_(0.8, 1.2)

            def lp(x):
                return lowpass_biquad(
                    x,
                    sample_rate=self._sr,
                    cutoff_freq=cutoff_freq * rand_factors[0],
                    Q=q_factor * rand_factors[1],
                )

        else:

            def lp(x):
                return lowpass_biquad(
                    x, sample_rate=self._sr, cutoff_freq=cutoff_freq, Q=q_factor
                )

        def reverse(x):
            return torch.flip(input=x, dims=[1])

        self._audio = reverse(lp(reverse(lp(self._audio))))

        # un-pad
        self._audio = self._audio[..., self.padding_length : -self.padding_length]

    def normalize(self, percent=0.99999):
        """
         Map audio values to [-1,1] and cut extremes values

        Args:
            percent: float,
                the percentage of values than will be kept before linear mapping,
                others are assigned to max or min

        """

        sorted_audio = torch.sort(abs(self._audio))[
            0
        ].squeeze()  # values of sorted audio
        cut = int(torch.numel(self._audio) * percent)
        new_abs_max = sorted_audio[cut]

        self._audio[self._audio > new_abs_max] = new_abs_max
        self._audio[self._audio < -new_abs_max] = -new_abs_max

        a = 1 / new_abs_max if new_abs_max != 0 else 0
        self._audio = a * self._audio

    def resampling(self, new_freq: int):
        """
        Resample the signal to new_freq

            Args:
                new_freq:int,
                    new_freq of the signal
        """

        resampling = Resample(
            orig_freq=self._sr, resampling_method="kaiser_window", new_freq=new_freq
        )
        self._audio = resampling(self._audio)
        self._sr = new_freq

    def select_part(self, len_seconds: float):
        """
        Select a part of a signal

        Args:
            len_seconds: duration of selected signal

        """
        real_time_len = self._audio.shape[1]
        desired_time_len = int(len_seconds * self._sr)
        if real_time_len >= desired_time_len:  # cut tensor
            if not self.determinist:
                start_idx = torch.randint(
                    low=0, high=real_time_len - desired_time_len + 1, size=(1,)
                )
            else:
                start_idx = int((real_time_len - desired_time_len) / 4)
            self._audio = self._audio[:, start_idx : start_idx + desired_time_len]
        else:
            self._audio = torch.cat(
                [self._audio, torch.zeros(1, desired_time_len - real_time_len)], dim=1
            )
