# EBEN: Extreme Bandwidth Extension Network applied to speech signals captured with noise-resilient microphones 

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>

This repository is the official implementation of [EBEN](https://arxiv.org/abs/2210.14090). Please visit the [project page](https://jhauret.github.io/eben/) !

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Obtain your trained EBEN model

### Option 1: download the pre-trained model discussed in [the article](https://arxiv.org/abs/2210.14090)
```bash
wget https://acoustique.cnam.fr/eben/last.ckpt
```
### Option 2: train your own model from scratch
#### Download [French Librispeech](https://arxiv.org/abs/2012.03411)

```bash
wget https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz
tar xvf mls_french.tar.gz
rm mls_french.tar.gz
```
#### Training

To train EBEN, run this command:

```train
python train.py
```

## Evaluation

To evaluate EBEN on French Librispeech, run:

```eval
python test.py
```
## Results

Our model achieves the following performance on :

### [Bandwidth Extension](https://paperswithcode.com/task/bandwidth-extension)

| Speech\Metrics | PESQ               | SI-SDR              | STOI                 | MUSHRA-I         | MUSHRA-Q         | Gen params     | Dis params      |
|------------------------------------------------------|--------------------|---------------------|----------------------|------------------|------------------|----------------|-----------------|
| Simulated In-ear                                     | 2.42 (0.34)        | 8.4 (3.7)           | 0.83 (0.05)          | 51 (29)          | 24 (18)          | $\emptyset$    | $\emptyset$     |
| Audio U-net                 | **2.24 (0.49)** | **11.9 (3.7)** | 0.87 (0.04)          | 60 (26)          | 33 (18)          | 71.0 M         | $\emptyset$     |
| Hifi-GAN v3                      | 1.32 (0.16)        | -25.1 (11.4)        | 0.78 (0.04)          | 40 (23)          | 36 (18)          | 1.5 M          | 70.7 M          |
| Seanet                 | 1.92 (0.48)        | 11.1 (3.0)          | **0.89 (0.04)** | **73 (13)** | **78 (12)** | 8.3 M          | 56.6  M         |
| Streaming-Seanet                        | 2.01 (0.46)        | 11.2 (3.6)          | **0.89 (0.04)** | 66 (20)          | 61 (14)          | **0.7 M** | 56.6  M         |
| EBEN (ours)                                          | 2.08 (0.45)        | 10.9 (3.3)          | **0.89 (0.04)** | **73 (14)** | **76 (14)** | 1.9 M          | **26.5 M** |

## Cite our work

```
@article{hauret2023eben,
        title={EBEN: Extreme Bandwidth Extension Network applied to
        speech signals captured with noise-resilient microphones},
        author={Hauret, Julien and Joubaud, Thomas and Zimpfer,
               Véronique and Bavu, Eric},
        journal={Submitted to ICASSP 2023}
        doi = {10.48550/ARXIV.2210.14090},
        year = {2022},
        url = {https://arxiv.org/abs/2210.14090},
        copyright = {Creative Commons Attribution 4.0 International}}
```
