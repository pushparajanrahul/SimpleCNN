# SimpleCNN
This repository contains a simple Convolutional Neural Network (CNN) implemented in PyTorch for classifying the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9), and the goal is to train a CNN model to correctly classify these digits.

NOTE : If you couldn't understand any terminology given below, please take some time to Google the definition in the context of ML/DL, for a better clarity.

# Lets Talk about Dataset - The MNIST Dataset

The MNIST dataset primarily consist of about 70,000 handwritten numeric images and their respective labels marked to it. Out of the total, 60,000 images are that of training images and 10,000 are test images, 
all of which are of the dimension 28pixel by 28pixel. 

MNIST dataset is monochromatic, featuring a sole color channel, in contrast to the RGB color format, which makes the number of channels as 1 instead on 3 (for RGB images).

General site for the MNIST database: http://yann.lecun.com/exdb/mnist

Below is a sample of MNIST Dataset. 

![MNIST_Sample](https://github.com/pushparajanrahul/SimpleCNN/assets/124497777/0e379fa2-dae6-43e6-8bb8-00fc29a7e99b)

# Prerequisites

Before running the code, make sure you have the following libraries installed:

PyTorch
torchvision
matplotlib

You can install these libraries using pip:



```bash
pip install torch torchvision matplotlib
`\```bash




