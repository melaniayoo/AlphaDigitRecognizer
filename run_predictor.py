import torch
from drawing_predictor import DrawingApp
import tkinter as tk
import sys
from torchvision import datasets
from test_EMNIST import CNN_EMNIST  # Import CNN for EMNIST
from test_MNIST import CNN_MNIST  # Import CNN for MNIST

emnist_letters_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Function to load the model
def load_model(dataset_choice):
    if dataset_choice == 'EMNIST':
        # Load the EMNIST model
        model = CNN_EMNIST(out_1=16, out_2=32, num_classes=26)  # For EMNIST, there are 26 classes
        model.load_state_dict(torch.load('emnist_letters_model.pth'))  # Load EMNIST model weights
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

    # Pass the model and dataset information to the drawing app
    class_mapping = emnist_letters_mapping if dataset_choice == "EMNIST" else None
    app = DrawingApp(root, model=model, dataset_choice=dataset_choice, class_mapping=class_mapping)

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
