# Handwritten Character Recognition with EMNIST (PyTorch)

This repository contains a clean, end-to-end example of handwritten character recognition
using the **EMNIST Letters** dataset and a simple convolutional neural network (CNN) written in PyTorch.

The goal is to classify 28×28 grayscale images of handwritten English letters A–Z.

## Project overview

- **Task**: multi-class image classification (26 classes)
- **Dataset**: [EMNIST Letters] – an extension of MNIST that includes letters as well as digits
- **Model**: small CNN trained from scratch
- **Framework**: PyTorch + Torchvision

The code is written to be easy to understand and extend for your own experiments.

## Dataset

We use the *Letters* split of EMNIST, which merges upper- and lower-case characters into a single
26-class problem. The dataset is automatically downloaded by `torchvision.datasets.EMNIST`, so no
manual download is required.

By default, the images are rotated; we rotate and flip them back to a readable orientation in
`src/data_utils.py`.



## Evaluation
After training, evaluate the model on the test set:


python evaluate.py --checkpoint models/emnist_letters_cnn_best.pth
This prints a classification report and writes a confusion matrix to:


outputs/figures/confusion_matrix.png

Possible extensions
Some ideas to extend this project:

Add data augmentation (random affine transforms, elastic distortions)

Try deeper architectures (ResNet, WideResNet, etc.)

Compare different EMNIST splits: balanced, digits, byclass

Export the model to ONNX / TorchScript

Build a small web interface where users draw a letter and the model predicts it
