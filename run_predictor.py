import torch
from drawing_predictor import DrawingApp
import tkinter as tk
import sys
from torchvision import datasets
from test_EMNIST import CNN_EMNIST  # Import CNN for EMNIST
from test_MNIST import CNN_MNIST  # Import CNN for MNIST


# Function to load the model
def load_model(dataset_choice):
    if dataset_choice == 'EMNIST':
        # Load the EMNIST model
        model = CNN_EMNIST(out_1=16, out_2=32, num_classes=47)  # For EMNIST, there are 47 classes
        model.load_state_dict(torch.load('emnist_model.pth'))  # Load EMNIST model weights
    elif dataset_choice == 'MNIST':
        # Load the MNIST model
        model = CNN_MNIST(out_1=16, out_2=32, num_classes=10)  # For MNIST, there are 10 classes
        model.load_state_dict(torch.load('mnist_model.pth'))  # Load MNIST model weights
    else:
        raise ValueError("Invalid dataset choice. Choose either 'EMNIST' or 'MNIST'")
    
    model.eval()  # Set the model to evaluation mode
    return model

# Function to initialize the application with the selected model
def run_drawing_predictor(dataset_choice):
    model = load_model(dataset_choice)
    
    # Initialize the Tkinter window and application
    root = tk.Tk()
    root.title(f"{dataset_choice} Drawing Predictor")
    
    # Modify the DrawingApp class to use the correct model
    app = DrawingApp(root, model=model)
    
    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    # Take the dataset choice as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python run_predictor.py <EMNIST or MNIST>")
        sys.exit(1)
    
    dataset_choice = sys.argv[1]
    
    # Run the drawing predictor with the selected dataset
    run_drawing_predictor(dataset_choice)
