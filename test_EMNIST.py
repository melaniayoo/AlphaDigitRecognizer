import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

# Function to display a data sample
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('Label = ' + str(data_sample[1]))

# Constants
IMAGE_SIZE = 28  # EMNIST Letters images are 28x28
NUM_CLASSES = 26  # A-Z letters

# Transformations: Resize, ToTensor, Normalize
composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])

# Load EMNIST Letters dataset
train_dataset = dsets.EMNIST(
    root='./data', 
    split='letters',  # Use 'letters' split for 26-class data
    train=True, 
    download=True, 
    transform=composed
)

validation_dataset = dsets.EMNIST(
    root='./data', 
    split='letters', 
    train=False, 
    download=True, 
    transform=composed
)

train_dataset[3][1]
show_data(train_dataset[3])

# Define the CNN model
class CNN_EMNIST(nn.Module):
    def __init__(self, out_1=16, out_2=32, num_classes=NUM_CLASSES):
        super(CNN_EMNIST, self).__init__()
        self.cnn1 = nn.Conv2d(1, out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(out_1, out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 7 * 7, num_classes)  # 7x7 after pooling twice
    
    def forward(self, x):
        x = torch.relu(self.cnn1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.cnn2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Instantiate the model, loss function, and optimizer
model = CNN_EMNIST(out_1=16, out_2=32, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1000, shuffle=False)

# Training Function
def train_model(n_epochs):
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y - 1)  # EMNIST Letters labels start at 1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x_val, y_val in validation_loader:
                y_pred = model(x_val)
                _, yhat = torch.max(y_pred, 1)
                correct += (yhat == (y_val - 1)).sum().item()
        
        accuracy = correct / len(validation_dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Train and save the model
n_epochs = 15
train_model(n_epochs)
torch.save(model.state_dict(), 'emnist_letters_model.pth')
