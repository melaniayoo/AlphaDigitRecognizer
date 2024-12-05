import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

# Function to display sample data
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))
    plt.show()

# EMNIST DATASET
IMAGE_SIZE = 16  # Resize to 16x16

# Define transformations
composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Load EMNIST `letters` dataset
train_dataset = dsets.EMNIST(
    root='./data', 
    split='letters',  # Use `letters` split for A-Z
    train=True, 
    download=True, 
    transform=composed
)

validation_dataset = dsets.EMNIST(
    root='./data', 
    split='letters', 
    split='letters', 
    train=False, 
    download=True, 
    transform=composed
)

# Define a wrapper to adjust dataset labels
class AdjustLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label - 1  # Adjust labels to be 0–25

# Wrap datasets to adjust labels
train_dataset = AdjustLabelsDataset(train_dataset)
validation_dataset = AdjustLabelsDataset(validation_dataset)

# Define the CNN Model
class CNN_EMNIST(nn.Module):
    def __init__(self, out_1=16, out_2=32, num_classes=26):  # Adjust num_classes for 26 classes
        super(CNN_EMNIST, self).__init__()
        # Convolutional layers
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Calculate the flattened size dynamically
        dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)  # Adjust input size if needed
        x = self.cnn1(dummy_input)
        x = self.maxpool1(torch.relu(x))
        x = self.cnn2(x)
        x = self.maxpool2(torch.relu(x))
        self.flattened_size = x.view(-1).size(0)

        # Fully connected layer
        self.fc1 = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = torch.relu(self.cnn2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Initialize the CNN model for EMNIST `letters` split (26 classes)
model = CNN_EMNIST(out_1=16, out_2=32, num_classes=26)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Model Training Function
n_epochs = 3  # Number of epochs
cost_list = []  # To track loss
accuracy_list = []  # To track accuracy

def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        model.train()  # Set model to training mode
        for x, y in train_loader:
            optimizer.zero_grad()  # Reset gradients
            z = model(x)  # Forward pass
            loss = criterion(z, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            COST += loss.item()  # Accumulate loss

        # Validation accuracy
        correct = 0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for x_val, y_val in validation_loader:
                z = model(x_val)  # Forward pass
                _, yhat = torch.max(z, 1)  # Get predictions
                correct += (yhat == y_val).sum().item()

        accuracy = correct / len(validation_dataset)
        cost_list.append(COST)
        accuracy_list.append(accuracy)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {COST:.4f}, Accuracy: {accuracy:.4f}")

# Train the model
train_model(n_epochs)

# Save the trained model
torch.save(model.state_dict(), 'emnist_letters_model.pth')
print("Model saved as emnist_letters_model.pth")
