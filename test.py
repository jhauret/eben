""" Python script to test EBEN model """

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from src.eben import EBEN
from src.generator import GeneratorEBEN
from src.librispeech_datamodule import CustomLibriSpeechDM
from torchmetrics import MetricCollection, ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import (PerceptualEvaluationSpeechQuality,
                                ShortTimeObjectiveIntelligibility)


def test():
    """actual testing function"""

    # Instantiate datamodule
    datamodule: LightningDataModule = CustomLibriSpeechDM(
        path_to_dataset="./mls_french",
        sr_standard=16000,
        bs_train=16,
        len_seconds_train=2,
        num_workers=4,
    )

    # Instantiate test metrics (STOI and PESQ are long to compute: CPU only)
    metrics = MetricCollection(
        {
            "si_sdr": ScaleInvariantSignalDistortionRatio(),
            "stoi": ShortTimeObjectiveIntelligibility(fs=16000),
            "pesq": PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),
        }
    )

    # Instantiate EBEN and load pre-trained weights
    generator: torch.nn.Module = GeneratorEBEN(m=4, n=32, p=1)

    weights = torch.load(
        "./generator.ckpt"
    )  # or 'generator_retrained.ckpt' if you have previously run train.py

    generator.load_state_dict(weights)

    eben: LightningModule = EBEN(generator=generator, metrics=metrics)

    trainer: Trainer = Trainer(gpus=1, logger=False)

    # Test
    trainer.test(model=eben, datamodule=datamodule)


if __name__ == "__main__":
    test()
