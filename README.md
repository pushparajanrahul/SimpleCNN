# SimpleCNN
This repository contains a simple CNN model, trained on MNIST Dataset, and this excercie is actively done by a lot of DL enthusiasts/begineers and is even considered as an indroductory excercise for Image processing courses. I am lending the same concept but I will try to comprehend the example with step by step guidance on how to build the algorithm i.e., the Simple CNN architecture and utilize it for training a Dataset (MNIST Dataset in this case). 

NOTE : If you couldn't understand any terminology given below, please take some time to Google the definition in the context of ML/DL, for a better clarity.

# Lets Talk about Dataset - The MNIST Dataset

The MNIST dataset primarily consist of about 70,000 handwritten numeric images and their respective labels marked to it. Out of the total, 60,000 images are that of training images and 10,000 are test images, 
all of which are of the dimension 28pixel by 28pixel. 

MNIST dataset is monochromatic, featuring a sole color channel, in contrast to the RGB color format, which makes the number of channels as 1 instead on 3 (for RGB images).

General site for the MNIST database: http://yann.lecun.com/exdb/mnist

Numerous renowned machine learning algorithms have been applied to the MNIST dataset, making it straightforward to evaluate the relative performance of a new algorithm. In December 2011, the website http://yann.lecun.com/exdb/mnist/ was updated to provide a comprehensive list of major classification techniques and their corresponding results achieved using the MNIST dataset. In most of these experiments, the classifiers were trained using the existing data from the MNIST database, as indicated by "none" in the "Preprocessing" column on the website. In some instances, the training set was enhanced by incorporating artificially altered versions of the original training samples. These alterations involved random combinations of jittering, shifts, scaling, deskewing, deslanting, blurring, and compression. The specific types of these and other modifications are specified in the "Preprocessing" column of the table.

Below is a sample of MNIST Dataset. 

![MNIST_Sample](https://github.com/pushparajanrahul/SimpleCNN/assets/124497777/0e379fa2-dae6-43e6-8bb8-00fc29a7e99b)
