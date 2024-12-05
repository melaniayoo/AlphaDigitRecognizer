import torch
from drawing_predictor import DrawingApp
import tkinter as tk
import sys
from torchvision import datasets
from test_EMNIST import CNN_EMNIST  # Import CNN for EMNIST
from test_MNIST import CNN_MNIST  # Import CNN for MNIST

# Mapping for EMNIST Balanced split (class index to character)
emnist_class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
    40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k'
}

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
    app = DrawingApp(root, model=model, dataset_choice=dataset_choice)
    
    # Run the Tkinter event loop
    root.mainloop()

# Function to predict class (letter) from the image
def predict_class(model, image_tensor, dataset_choice):
    """Given a model and an image tensor, predict the class and return the corresponding letter."""
    with torch.no_grad():
        output = model(image_tensor)
    # Get the predicted class index (the class with the max value)
    _, predicted_class_idx = torch.max(output, 1)
    
    # Map the class index to the corresponding letter based on the dataset
    if dataset_choice == 'EMNIST':
        predicted_class_letter = EMNIST_CLASSES[predicted_class_idx.item()]
    else:
        # For MNIST, there is no need for mapping as it only has digits (0-9)
        predicted_class_letter = str(predicted_class_idx.item())
    
    return predicted_class_letter

if __name__ == "__main__":
    # Take the dataset choice as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python run_predictor.py <EMNIST or MNIST>")
        sys.exit(1)
    
    dataset_choice = sys.argv[1]
    
    # Run the drawing predictor with the selected dataset
    run_drawing_predictor(dataset_choice)
