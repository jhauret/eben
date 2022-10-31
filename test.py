
""" Python script to test EBEN model """

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics import MetricCollection, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality

from eben import EBEN
from generator import GeneratorEBEN
from discriminator import DiscriminatorEBENMultiScales
from librispeech_datamodule import CustomLibriSpeechDM


def test():
    """   actual testing function    """

    # Instantiate datamodule
    datamodule: LightningDataModule = CustomLibriSpeechDM(path_to_dataset='./mls_french',
                                                          sr_standard=16000, bs_train=16,
                                                          len_seconds_train=2, num_workers=4)

    # Instantiate test metrics (STOI and PESQ are long to compute: CPU only)
    metrics = MetricCollection({'si_sdr': ScaleInvariantSignalDistortionRatio(),
                                'stoi': ShortTimeObjectiveIntelligibility(fs=16000),
                                'pesq': PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')})

    # Instantiate EBEN
    generator: torch.nn.Module = GeneratorEBEN(bands_nbr=4, pqmf_ks=32)
    discriminator: torch.nn.Module = DiscriminatorEBENMultiScales()
    eben: LightningModule = EBEN(generator=generator, discriminator=discriminator, metrics=metrics)
    trainer: Trainer = Trainer(gpus=1)

    # Test
    ckpt = './lightning_logs/version_0/checkpoints/epoch=34-step=1129660.ckpt'
    trainer.test(model=eben, datamodule=datamodule, ckpt_path=ckpt)


if __name__ == '__main__':

    test()
