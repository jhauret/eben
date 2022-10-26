# EBEN: Extreme bandwidth extension network applied to speech signals captured with noise-resilient microphones 	

This repository is the official implementation of [EBEN](https://arxiv.org/abs/2210.14090). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
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
