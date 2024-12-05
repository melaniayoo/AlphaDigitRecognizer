# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import os

# Constants
IMAGE_SIZE = 16  # Resize to 16x16
NUM_CLASSES = 26  # A-Z (uppercase letters only)

# Define transformations
compose_rotate = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(45),  # Random rotation up to 45 degrees
    transforms.ToTensor()
])

compose = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Load EMNIST `letters` dataset
train_dataset = dsets.EMNIST(
    root='./data',
    split='letters',
    train=True,
    download=True,
    transform=compose
)

validation_dataset = dsets.EMNIST(
    root='./data',
    split='letters',
    train=False,
    download=True,
    transform=compose_rotate  # Apply rotation to validation data
)

# Adjust dataset labels (convert 1-26 to 0-25 for compatibility)
class AdjustLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label - 1  # Adjust labels to 0-25

# Wrap datasets to adjust labels
train_dataset = AdjustLabelsDataset(train_dataset)
validation_dataset = AdjustLabelsDataset(validation_dataset)

# CNN Model Definition
class CNN_EMNIST_ROTATE(nn.Module):
    def __init__(self, out_1=16, out_2=32, num_classes=NUM_CLASSES):
        super(CNN_EMNIST_ROTATE, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Calculate the flattened size dynamically
        dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = self.cnn1(dummy_input)
        x = self.maxpool1(torch.relu(x))
        x = self.cnn2(x)
        x = self.maxpool2(torch.relu(x))
        self.flattened_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Initialize Model
model = CNN_EMNIST_ROTATE(out_1=16, out_2=32, num_classes=NUM_CLASSES)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Training Function
def train_model(model, train_loader, validation_loader, n_epochs, save_path):
    checkpoint = {
        'epoch': None,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'loss': None,
        'cost': [],
        'accuracy': []
    }

    N_test = len(validation_loader.dataset)

    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        # Training Loop
        for x, y in train_loader:
            optimizer.zero_grad()  # Reset gradients
            z = model(x)  # Forward pass
            loss = criterion(z, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Accumulate loss

        # Validation Loop
        model.eval()  # Set model to evaluation mode
        correct = 0
        with torch.no_grad():
            for x_val, y_val in validation_loader:
                z = model(x_val)  # Forward pass
                _, yhat = torch.max(z, 1)  # Get predictions
                correct += (yhat == y_val).sum().item()

        accuracy = correct / N_test
        checkpoint['epoch'] = epoch + 1
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['loss'] = total_loss
        checkpoint['cost'].append(total_loss)
        checkpoint['accuracy'].append(accuracy)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the trained model and checkpoint
    torch.save(checkpoint, save_path)
    print(f"Model and training data saved to {save_path}")

# Train the model
n_epochs = 10
save_path = os.path.join(os.getcwd(), 'emnist_letters_rotated_checkpoint.pth')
train_model(model, train_loader, validation_loader, n_epochs, save_path)
