import torch
from drawing_predictor import DrawingApp
import tkinter as tk
import sys
from test_EMNIST import CNN_EMNIST  # Import EMNIST model
from test_MNIST import CNN_MNIST  # Import MNIST model
import os

# Function to load the model
def load_model(dataset_choice):
    model_path = ""
    if dataset_choice == 'EMNIST':
        # Use the updated EMNIST model if applicable
        model = CNN_EMNIST(out_1=32, out_2=64, num_classes=26)  # Adjusted architecture
        model_path = 'Models/emnist_letters_model.pth'
    elif dataset_choice == 'MNIST':
        model = CNN_MNIST(out_1=16, out_2=32, num_classes=10)
        model_path = 'Models/mnist_model.pth'
    else:
        raise ValueError("Invalid dataset choice. Choose either 'EMNIST' or 'MNIST'.")

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to initialize the application with the chosen model
def run_drawing_predictor(dataset_choice):
    try:
        model = load_model(dataset_choice)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Initialize the Tkinter window and application
    root = tk.Tk()
    root.title(f"{dataset_choice} Drawing Predictor")

    # Launch the drawing application
    app = DrawingApp(root, model=model, dataset_choice=dataset_choice)

    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    # Ensure the dataset choice is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python run_predictor.py <EMNIST or MNIST>")
        sys.exit(1)
    
    dataset_choice = sys.argv[1].upper()  # Convert to uppercase for consistency
    if dataset_choice not in ['EMNIST', 'MNIST']:
        print("Invalid dataset choice. Choose either 'EMNIST' or 'MNIST'.")
        sys.exit(1)

    # Run the drawing predictor application
    run_drawing_predictor(dataset_choice)
