import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

#Training the CNN model
def train(num_epochs, cnn, loaders):
    cnn.to(device)
    cnn.train()
    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images = images.to(device)
            labels = labels.to(device)
            #b_x = Variable(images) # batch x
            #b_y = Variable(labels) # batch y
            output, x = cnn(images)  
            loss = loss_func(output, labels)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        pass
    pass



#Evaluating the model
def test():
    # Test the model
    cnn.eval()
    cnn.to(device)

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in loaders['test']:
            images = images.to(device) #Move images to CPU or GPU
            labels = labels.to(device) #Move images to CPU or GPU
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / float(total)

    print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


#Define the Convolutional Neural Network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, ), nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x # return x for visualization
    

if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name()
    print("The CNN model is moved to", device,"with device name as ", device_name)

    #Download MNIST dataset in local system

    train_data = datasets.MNIST(root = 'data',  train = True, transform = ToTensor(), download = True,)
    test_data = datasets.MNIST(root = 'data', train = False, transform = ToTensor())

    #Print train_data and test_data size
    print(train_data)
    print(test_data)
    print(train_data.data.size())
    print(train_data.targets.size())

    #Visualization of MNIST dataset
    
    plt.imshow(train_data.data[0], cmap='gray')
    plt.title('%i' % train_data.targets[0])
    plt.show()

    #Plot multiple train_data
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1) :
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

    #Preparing data for training with DataLoaders
    
    loaders = {
        'train' : DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        'test' : DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
    }
    #loaders


    #Printing the convolution meural network model
    cnn = CNN()
    print(cnn)
    

    #Define loss function
    loss_func = nn.CrossEntropyLoss()
    #loss_func


    #Define a Optimization Function
    
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
    #optimizer

    #Train the model
    num_epochs = 10
    start = time.time()
    train(num_epochs, cnn, loaders)
    stop = time.time()
    print(f"Total time :{stop-start}")



    # Evaluate the model on test data
    test()

    #Print 10 predictions from test data
    sample = next(iter(loaders['test']))
    imgs, lbls = sample
    actual_number = lbls[:10].numpy()
    #actual_number

    test_output, last_layer = cnn(imgs[:10].to(device))
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()#cpu() added because numpy does not support GPU
    print(f'Prediction number: {pred_y}')
    print(f'Actual number: {actual_number}')