# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt

# ASCII Mapping for EMNIST Balanced Dataset
EMNIST_LABELS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcedfghijklmnopqrstuvwxyz'

# Map the label `y` to its corresponding character
def map_label_to_char(label):
    return EMNIST_LABELS[label]

# Example: Convert label 20
print("Character for label 20:", map_label_to_char(20))

# Plotting function
# Updated function to show data with the mapped character
def show_data(data_sample):
    label = data_sample[1]
    character = map_label_to_char(label)
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title(f'y = {label}, char = {character}')
    plt.show()


# Load EMNIST dataset
train_dataset = dsets.EMNIST(
    root='./data', 
    split='balanced',  # Use 'balanced' for alphanumeric data
    train=True, 
    download=True, 
    transform=transforms.ToTensor()
)

validation_dataset = dsets.EMNIST(
    root='./data', 
    split='balanced', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor()
)

print("Print the training dataset:\n", train_dataset)

# Define SoftMax Classifier
class SoftMax(nn.Module):
    def __init__(self, input_size, output_size):
        super(SoftMax, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# Set input and output dimensions
input_dim = 28 * 28
output_dim = 47  # Adjust this to match the number of classes in 'balanced'

# Initialize the model
model = SoftMax(input_dim, output_dim)
print("Model:\n", model)

# Define optimizer, criterion, and data loaders
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

# Training loop
def train_model(n_epochs):
    for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        correct = 0
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, 28 * 28))
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / len(validation_dataset)
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}")

# Train the model
train_model(n_epochs=10)

# Display some predictions
Softmax_fn = nn.Softmax(dim=1)
count = 0
for x, y in validation_dataset:
    z = model(x.view(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat == y:
        show_data((x, y))
        plt.show()
        print("Prediction:", yhat.item())
        print("Probability:", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break

# Show another sample by changing the index
show_data(train_dataset[10])  # For index 10
show_data(train_dataset[20])  # For index 20
show_data(train_dataset[100])  # For index 100
