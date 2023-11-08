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
```

# Dataset and Dataloaders
The MNIST dataset is automatically downloaded and utilized for training and testing. The dataset details are as follows:

```cshrap
Training Dataset:
Number of datapoints: 60,000
Root location: data
Split: Train
Data Transformation: ToTensor()
Testing Dataset:
Number of datapoints: 10,000
Root location: data
Split: Test
Data Transformation: ToTensor()
```


# Code Structure

## 1. CNN Class
The CNN class defines the architecture of the Convolutional Neural Network. It consists of two convolutional layers followed by max-pooling and a fully connected layer. The network is designed to classify the input images into one of the ten possible digits (0-9).

The CNN architecture defined in the code is as follows:

```css
CNN(
  (conv1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (out): Linear(in_features=1568, out_features=10, bias=True)
)
```


## 2. Train Function
The train function is responsible for training the CNN model. It takes the number of training epochs, the model, and the data loaders as input. During training, the function iterates through the training data, calculates the loss, and updates the model's weights. It prints the training progress, including the current epoch and batch loss.

## 3. Test Function
The test function evaluates the trained model on the test data. It calculates the accuracy of the model on the test set and prints the test accuracy.

## 4. Main Program
The main program sets up the data loaders, initializes the model, loss function, and optimizer, and then trains the model for a specified number of epochs. After training, it evaluates the model on the test data and prints the predicted labels for a sample of test images.

# Usage
To run the code, execute the main section of the script. You can specify the number of training epochs by changing the num_epochs variable. The code will automatically download the MNIST dataset and display some sample images. After training, it will print the test accuracy and some example predictions.

# Training the Model
You can adjust the hyperparameters like learning rate, batch size, and network architecture in the CNN class or during the initialization of the optimizer. Feel free to experiment with different settings to improve the model's performance.

The model is trained over 10 epochs with a total training time of approximately 47.9 seconds. During training, the model achieves a high accuracy on the test data. The training progress is displayed with epoch and batch-wise loss.

# Testing the model
The model is evaluated on the test data, and it achieves a test accuracy of 99%.

# Sample Prediction
To validate the model's performance, 10 sample predictions are compared to their actual labels. The predictions closely match the actual digits, demonstrating the model's effectiveness in classifying handwritten digits.
