
""" Python script to train EBEN model """

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from eben import EBEN
from generator import GeneratorEBEN
from discriminator import DiscriminatorEBENMultiScales
from librispeech_datamodule import CustomLibriSpeechDM


def train():
    """   actual training function    """

    # Instantiate datamodule
    datamodule: LightningDataModule = CustomLibriSpeechDM(path_to_dataset='./mls_french',
                                                          sr_standard=16000, bs_train=8,
                                                          len_seconds_train=2, num_workers=4)

    # Instantiate EBEN
    generator: torch.nn.Module = GeneratorEBEN(bands_nbr=4, pqmf_ks=32)
    discriminator: torch.nn.Module = DiscriminatorEBENMultiScales()
    eben: LightningModule = EBEN(generator=generator, discriminator=discriminator,
                                 lr=0.0003, betas=(0.5, 0.9))
    trainer: Trainer = Trainer(gpus=1)

    # Fit
    trainer.fit(model=eben, datamodule=datamodule)


if __name__ == '__main__':

    train()
