# Updated test_EMNIST.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from torch.optim import Adam

# Function to display sample data
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))
    plt.show()

# EMNIST DATASET
IMAGE_SIZE = 16  # Resize to 16x16

# Define transformations
train_composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
])

val_composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load EMNIST `letters` dataset
train_dataset = dsets.EMNIST(
    root='./data', 
    split='letters', 
    train=True, 
    download=True, 
    transform=train_composed
)

validation_dataset = dsets.EMNIST(
    root='./data', 
    split='letters', 
    train=False, 
    download=True, 
    transform=val_composed
)

# Adjust labels
class AdjustLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label - 1  # Adjust labels to 0–25

train_dataset = AdjustLabelsDataset(train_dataset)
validation_dataset = AdjustLabelsDataset(validation_dataset)

# Define the CNN Model
class CNN_EMNIST(nn.Module):
    def __init__(self, out_1=32, out_2=64, num_classes=26):
        super(CNN_EMNIST, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(out_2 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# Initialize the CNN model for EMNIST
model = CNN_EMNIST(out_1=32, out_2=64, num_classes=26)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = Adam(model.parameters(), lr=learning_rate)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=256, shuffle=False)

# Training Function
n_epochs = 20
cost_list = []
accuracy_list = []

def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += loss.item()

        correct = 0
        model.eval()
        with torch.no_grad():
            for x_val, y_val in validation_loader:
                z = model(x_val)
                _, yhat = torch.max(z, 1)
                correct += (yhat == y_val).sum().item()

        accuracy = correct / len(validation_dataset)
        cost_list.append(COST)
        accuracy_list.append(accuracy)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {COST:.4f}, Accuracy: {accuracy:.4f}")

# Train and Save Model
train_model(n_epochs)
torch.save(model.state_dict(), 'Models/emnist_letters_model.pth')
