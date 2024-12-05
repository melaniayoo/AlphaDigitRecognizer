# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform data
import torchvision.transforms as transforms
# Allows us to download the dataset
import torchvision.datasets as dsets
# Used to graph data and loss curves
import matplotlib.pylab as plt
# Allows us to use arrays to manipulate and store data
import numpy as np

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))

# EMNIST DATASET
IMAGE_SIZE = 16

# First the image is resized then converted to a tensor
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# Load EMNIST dataset
train_dataset = dsets.EMNIST(
    root='./data', 
    split='balanced',  # Use 'balanced' for alphanumeric data
    train=True, 
    download=True, 
    transform=composed
)

validation_dataset = dsets.EMNIST(
    root='./data', 
    split='balanced', 
    train=False, 
    download=True, 
    transform=composed
)

class CNN_EMNIST(nn.Module):
    
    # Constructor
    def __init__(self, out_1=16, out_2=32, num_classes=47):
        super(CNN_EMNIST, self).__init__()
        # The reason we start with 1 channel is because we have a single black and white image
        # Channel Width after this layer is 16
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        # Channel Width after this layer is 8
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # Channel Width after this layer is 8
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        # Channel Width after this layer is 4
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Adjust output features to match number of classes in EMNIST (47 classes)
        self.fc1 = nn.Linear(out_2 * 4 * 4, num_classes)  # Adjust for EMNIST (47 classes)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each CNN, ReLU, and pooling layer, and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    # Outputs result of each stage of the CNN, ReLU, and pooling layers
    def activations(self, x):
        # Outputs activation (this is not necessary for training)
        z1 = self.cnn1(x)
        a1 = torch.relu(z1)
        out = self.maxpool1(a1)
        
        z2 = self.cnn2(out)
        a2 = torch.relu(z2)
        out1 = self.maxpool2(a2)
        out = out.view(out.size(0), -1)
        return z1, a1, z2, a2, out1, out

# Initialize the CNN model for EMNIST (47 classes)
model = CNN_EMNIST(out_1=16, out_2=32, num_classes=47)

# Create a criterion which will measure loss
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1

# Create an optimizer that updates model parameters using the learning rate and gradient
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Create a DataLoader for the training data with a batch size of 100 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)

# Create a DataLoader for the validation data with a batch size of 5000 
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

# Train the model

# Number of times we want to train on the training dataset
n_epochs = 3

# List to keep track of cost and accuracy
cost_list = []
accuracy_list = []

# Size of the validation dataset
N_test = len(validation_dataset)

# Model Training Function
def train_model(n_epochs):
    # Loops for each epoch
    for epoch in range(n_epochs):
        # Keeps track of cost for each epoch
        COST = 0
        # For each batch in train loader
        for x, y in train_loader:
            # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
            optimizer.zero_grad()
            # Makes a prediction based on X value
            z = model(x)
            # Measures the loss between prediction and actual Y value
            loss = criterion(z, y)
            # Calculates the gradient value with respect to each weight and bias
            loss.backward()
            # Updates the weight and bias according to calculated gradient value
            optimizer.step()
            # Cumulates loss 
            COST += loss.data
        
        # Saves cost of training data of epoch
        cost_list.append(COST)

        # Keeps track of correct predictions
        correct = 0
        # Perform a prediction on the validation data  
        for x_test, y_test in validation_loader:
            # Makes a prediction
            z = model(x_test)
            # The class with the max value is the one we are predicting
            _, yhat = torch.max(z.data, 1)
            # Checks if the prediction matches the actual value
            correct += (yhat == y_test).sum().item()
        
        # Calculates accuracy and saves it
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
     
train_model(n_epochs)
torch.save(model.state_dict(), 'emnist_model.pth')
