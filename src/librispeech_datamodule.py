""" Module to load Librispeech data and apply degradation """

import os
from pathlib import Path
from typing import Tuple

import torchaudio
from pytorch_lightning import LightningDataModule
from src.temporal_transforms import TemporalTransforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class CustomLibriSpeechDM(LightningDataModule):

    """
    Custom LibriSpeech LightningDataModule.

    This class encapsulates Pytorch dataloaders, prepare date and apply transforms.

    Args:
    -----------
        path_to_dataset: str, default None
            Folder that contains the dataset

        len_seconds_train: str, default None
            Sample length for train dataset in seconds

        len_seconds_val: str, default None
            Sample length for validation and test datasets in seconds

        bs_train: str, default None
            Batch size for train dataset

        bs_val: str, default None
            Batch size for validation and test datasets

        num_workers: int, default None
            Number of workers used to load data

        sr_standard: int, default True
            Sampling rate at which all samples are resample

        separator: str, default None
            ASCII character used in file names

        train_folder: str, default train/audio
            Folder that contains the train dataset

        val_folder: str, default dev/audio
            Folder that contains the validation dataset

        test_folder: str, default test/audio
            Folder that contains the test dataset

    """

    def __init__(
        self,
        path_to_dataset,
        sr_standard,
        len_seconds_train,
        bs_train,
        len_seconds_val=6,
        bs_val=8,
        num_workers=4,
        separator="_",
        train_folder="train/audio",
        val_folder="dev/audio",
        test_folder="test/audio",
    ):
        super().__init__()

        self.path_to_dataset = path_to_dataset
        self.bs_train = bs_train
        self.bs_val = bs_val
        self.num_workers = num_workers
        self.sr_standard = sr_standard
        self.len_seconds_train = len_seconds_train
        self.len_seconds_val = len_seconds_val
        self._separator = separator
        self._train_folder = train_folder
        self._val_folder = val_folder
        self._test_folder = test_folder

    def setup(self, stage=None) -> None:
        """
        Things to do on every accelerator in distributed mode
        """

        self.train_set = CustomLibriSpeechDS(
            path=self.path_to_dataset,
            folder=self._train_folder,
            deterministic=True,
            sr_standard=self.sr_standard,
            len_seconds=self.len_seconds_train,
            separator=self._separator,
        )
        self.val_set = CustomLibriSpeechDS(
            path=self.path_to_dataset,
            folder=self._val_folder,
            deterministic=True,
            sr_standard=self.sr_standard,
            len_seconds=self.len_seconds_val,
            separator=self._separator,
        )
        self.test_set = CustomLibriSpeechDS(
            path=self.path_to_dataset,
            folder=self._test_folder,
            deterministic=True,
            sr_standard=self.sr_standard,
            len_seconds=self.len_seconds_val,
            separator=self._separator,
        )

    def train_dataloader(self):
        """
        This function creates the train DataLoader.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.bs_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        This function creates the validation DataLoader.
        """
        return DataLoader(
            self.val_set,
            batch_size=self.bs_val,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        This function creates the test DataLoader.
        """
        return DataLoader(
            self.test_set,
            batch_size=self.bs_val,
            shuffle=False,
            num_workers=self.num_workers,
        )


class CustomLibriSpeechDS(Dataset):

    """Create a Dataset for LibriSpeech"""

    _ext_audio = ".flac"

    def __init__(
        self,
        path,
        folder,
        sr_standard,
        separator,
        len_seconds: float = 6.0,
        deterministic: bool = False,
    ) -> None:
        self.len_seconds = len_seconds
        self.sr_standard = sr_standard
        self.determinist = deterministic
        self._separator = separator
        self._path = path
        self._folder = folder
        self._full_path = os.path.join(self._path, self._folder)
        self._walker = sorted(
            str(p.stem) for p in Path(self._full_path).glob("*/*/*" + self._ext_audio)
        )

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        """Load the n-th sample from the dataset and apply degradation

        Args:
            n:int,
                The index of the sample to be loaded

        Returns:
            audio:torch.Tensor,
                audio is shape [1,time_len]
            audio_corrupted:torch.Tensor,
                audio_corrupted is shape [1,time_len]
        """
        fileid = self._walker[n]
        audio, sr = self.load_librispeech_item(fileid, self._full_path, self._ext_audio)

        tt_audio_ref = TemporalTransforms(audio, sr, deterministic=self.determinist)
        tt_audio_ref.resampling(new_freq=self.sr_standard)
        tt_audio_ref.select_part(self.len_seconds)
        tt_audio_corrupted = TemporalTransforms(tt_audio_ref.audio, tt_audio_ref.sr)
        tt_audio_corrupted.remove_hf()
        tt_audio_corrupted.add_noise()

        # smoothing boarders
        tt_audio_ref.smoothing()
        tt_audio_corrupted.smoothing()

        # normalize
        tt_audio_ref.normalize()
        tt_audio_corrupted.normalize()

        return tt_audio_ref.audio, tt_audio_corrupted.audio

    def __len__(self) -> int:
        return len(self._walker)

    def load_librispeech_item(
        self, fileid: str, path: str, ext_audio: str
    ) -> Tuple[Tensor, int]:
        speaker_id, chapter_id, utterance_id = fileid.split(self._separator)

        fileid_audio = (
            speaker_id + self._separator + chapter_id + self._separator + utterance_id
        )
        file_audio = fileid_audio + ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(
            file_audio, normalize=False
        )  # normalize

        return waveform, sample_rate
