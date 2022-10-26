# EBEN: Extreme Bandwidth Extension Network applied to speech signals captured with noise-resilient microphones 

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>

This repository is the official implementation of [EBEN](https://arxiv.org/abs/2210.14090). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Download data

#### [French Librispeech](https://arxiv.org/abs/2012.03411)

```bash
wget https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz
tar xvf mls_french.tar.gz
rm mls_french.tar.gz
```

#### [English Librispeech](https://ieeexplore.ieee.org/document/7178964)

```bash
comming soon...
```

## Training

To train EBEN, run this command:

```train
python train.py
```

## Evaluation

To evaluate EBEN on French Librispeech, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

You can download pretrained models here:

- [EBEN](https://drive.google.com/mymodel.pth) trained on French Librispeech with degradation x. 
- 

## Results

Our model achieves the following performance on :

### [Bandwidth Extension](https://paperswithcode.com/task/bandwidth-extension)

| Model name         |     MUSHRA-I    |    MUSHRA-Q    |
| ------------------ |---------------- | -------------- |
|   EBEN             |     73          |       76       |
