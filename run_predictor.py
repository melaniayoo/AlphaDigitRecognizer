import torch
from drawing_predictor import DrawingApp
import tkinter as tk
import sys
from test_EMNIST import CNN_EMNIST  # Import CNN for EMNIST
from test_MNIST import CNN_MNIST  # Import CNN for MNIST
from augmentation import CNN_EMNIST_ROTATE  # Import CNN for EMNIST with rotation

# Function to load the model
def load_model(dataset_choice):
    if dataset_choice == 'EMNIST':
        # Load the EMNIST model
        model = CNN_EMNIST_ROTATE(out_1=16, out_2=32, num_classes=26)  # For EMNIST, there are 26 classes
        model.load_state_dict(torch.load('emnist_letters_rotated_checkpoint.pth'))  # Load EMNIST model weights
    elif dataset_choice == 'MNIST':
        # Load the MNIST model
        model = CNN_MNIST(out_1=16, out_2=32, num_classes=10)  # For MNIST, there are 10 classes
        model.load_state_dict(torch.load('mnist_model.pth'))  # Load MNIST model weights
    else:
        raise ValueError("Invalid dataset choice. Choose either 'EMNIST' or 'MNIST'")
    
    model.eval()  # Set the model to evaluation mode
    return model

# Function to initialize the application with the EMNIST model
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
        predicted_class_letter = emnist_letters_mapping[predicted_class_idx.item()]
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
