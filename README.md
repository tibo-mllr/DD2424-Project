# DD2424-Project

## Authors

FAREEK SHEKH Bawar, GÖRANSSON Love, MULLER Thibault

## Description

This repository contains the code for the project of the course DD2424 Deep Learning in Data Science at KTH Royal Institute of Technology. The project is about the implementation of a deep learning model for the classification of the CIFAR-10 dataset with a convolutional network. The model is implemented in PyTorch.

## Usage

To train the model, run the following command:

```bash
python3 main.py
```

The model will be trained for 100 epochs and the training and validation losses and accuracies will be plotted at the end.

You can choose the model to train, and change the hyperparameters using the CLI. For example,

```bash
python3 main.py --model=batch-dropout --batch-size=128 --optimizer=Adam
```

will train the model with batch normalization followed by dropout, with a batch size of 128 and the Adam optimizer.

To see all available options, run:

```bash
python3 main.py -h
```

