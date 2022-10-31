
""" Definition of EBEN and its training pipeline with pytorch lightning paradigm"""

import torch
import pytorch_lightning as pl


class EBEN(pl.LightningModule):
    """
    EBEN LightningModule
    """

    def __init__(self, generator, discriminator=None, lr=None, betas=None, metrics=None):
        super().__init__()

        self.sr = 16000

        self.generator = generator
        self.discriminator = discriminator

        self.lr = lr
        self.betas = betas

        self.l1 = torch.nn.L1Loss()
        self.relu = torch.nn.ReLU()

        self.metrics = metrics

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        # batch is [audio_ref, audio_corrupted]
        cut_batch = [self.generator.cut_tensor(speech) for speech in batch]

        corrupted_speech = cut_batch[0]
        reference_speech = cut_batch[1]

        enhanced_speech, decomposed_enhanced_speech = self.generator(corrupted_speech)
        decomposed_reference_speech = self.generator.pqmf.forward(reference_speech, 'analysis')
        enhanced_embeddings = self.discriminator(bands=decomposed_enhanced_speech[:, 1:, :],
                                                 audio=enhanced_speech)
        reference_embeddings = self.discriminator(bands=decomposed_reference_speech[:, 1:, :],
                                                  audio=reference_speech)

        outs = {'reference': reference_speech, 'corrupted': corrupted_speech,
                'enhanced': enhanced_speech}

        # train generator
        if optimizer_idx == 0:

            # ftr_loss
            ftr_loss = 0
            for scale in range(len(reference_embeddings)):  # across scales
                for layer in range(1, len(reference_embeddings[scale]) - 1):  # across layers
                    a = reference_embeddings[scale][layer]
                    b = enhanced_embeddings[scale][layer]
                    ftr_loss += self.l1(a, b) / (len(reference_embeddings[scale]) - 2)
            ftr_loss /= len(reference_embeddings)

            # loss_adv_gen
            adv_loss = 0
            for scale in range(len(enhanced_embeddings)):  # across embeddings
                certainties = enhanced_embeddings[scale][-1]
                adv_loss += self.relu(1 - certainties).mean()  # across time
            adv_loss /= len(enhanced_embeddings)

            gen_loss = adv_loss + 100 * ftr_loss

            outs.update({'loss': gen_loss})

            return outs

        # train discriminator
        if optimizer_idx == 1:

            # valid_loss
            adv_loss_valid = 0
            for scale in range(len(reference_embeddings)):  # across embeddings
                certainties = reference_embeddings[scale][-1]
                adv_loss_valid += self.relu(1 - certainties).mean()  # across time
            adv_loss_valid /= len(reference_embeddings)

            # fake_loss
            adv_loss_fake = 0
            for scale in range(len(enhanced_embeddings)):  # across embeddings
                certainties = enhanced_embeddings[scale][-1]
                adv_loss_fake += self.relu(1 + certainties).mean()  # across time
            adv_loss_fake /= len(enhanced_embeddings)

            # loss to backprop on
            dis_loss = adv_loss_valid + adv_loss_fake

            # total_loss = ∑ losses
            outs.update({'loss': dis_loss})

        return outs

    def validation_step(self, batch, batch_idx):
        cut_batch = [self.generator.cut_tensor(speech) for speech in batch]

        reference_speech, corrupted_speech = cut_batch
        enhanced_speech, _ = self.generator(corrupted_speech)

        outs = {'reference': reference_speech, 'corrupted': corrupted_speech,
                'enhanced': enhanced_speech}

        return outs

    def test_step(self, batch, batch_idx):
        cut_batch = [self.generator.cut_tensor(speech) for speech in batch]

        reference_speech, corrupted_speech = cut_batch
        enhanced_speech, _ = self.generator(corrupted_speech)

        outs = {'reference': reference_speech, 'corrupted': corrupted_speech,
                'enhanced': enhanced_speech}

        return outs

    def configure_optimizers(self):

        optimizers = [
            torch.optim.Adam(params=self.generator.parameters(), lr=self.lr, betas=self.betas),
            torch.optim.Adam(params=self.discriminator.parameters(), lr=self.lr, betas=self.betas)]

        return optimizers

    def on_test_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        try:
            self.log_dict(self.metrics(outputs['enhanced'], outputs['reference']))
        except ValueError:
            print('ValueError')
