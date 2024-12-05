import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import tkinter as tk
import torch
from PIL import ImageGrab, Image
import numpy as np

# Load the TorchScript models
emnist_model = torch.jit.load("emnist_model.pt")
mnist_model = torch.jit.load("mnist_model.pt")

# Set both models to evaluation mode
emnist_model.eval()
mnist_model.eval()

# Variable to keep track of the selected model
current_model = "EMNIST"

# Function to map EMNIST/MNIST labels to characters
def label_to_char(label):
    if current_model == "EMNIST":
        # EMNIST mapping: A-Z starts from 0
        return chr(label + ord('A')) if label >= 10 else str(label)  # Map digits directly, letters afterward
    else:
        # MNIST mapping: 0-9 digits
        return str(label)

# Function to preprocess the canvas image
def preprocess_image(canvas):
    canvas.update()
    # Grab the canvas content as an image
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
    # Resize to 28x28, the input size for both models
    image = image.resize((16, 16), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    # Invert colors (black background, white foreground)
    image_array = 255 - image_array
    # Normalize and reshape for the model input
    image_array = image_array / 255.0  # Normalize pixel values
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return image_tensor

# Function to predict the character
def predict_character(canvas):
    global current_model
    # Preprocess the drawn image
    image = preprocess_image(canvas)

    # Select the appropriate model
    model = emnist_model if current_model == "EMNIST" else mnist_model

    # Predict the probabilities for each label
    with torch.no_grad():
        prediction = model(image)
        predicted_label = torch.argmax(prediction).item()
        confidence = torch.softmax(prediction, dim=1).max().item() * 100  # Confidence percentage

    # Map label to character
    predicted_char = label_to_char(predicted_label)
    result_label.config(text=f"Predicted: {predicted_char} (Confidence: {confidence:.2f}%)")

# Function to clear the canvas
def clear_canvas(canvas):
    canvas.delete("all")

# Function to switch models
def switch_model(selected_model):
    global current_model
    current_model = selected_model
    model_label.config(text=f"Current Model: {current_model}")

# Create the Tkinter GUI
root = tk.Tk()
root.title("Handwritten Character Recognition")

# Create a canvas for drawing
canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack()

# Add result display
result_label = tk.Label(root, text="Predicted: ", font=("Helvetica", 16))
result_label.pack()

# Add a label to display the current model
model_label = tk.Label(root, text=f"Current Model: {current_model}", font=("Helvetica", 14))
model_label.pack()

# Add buttons for model selection
model_frame = tk.Frame(root)
model_frame.pack()

emnist_button = tk.Button(model_frame, text="Use EMNIST", command=lambda: switch_model("EMNIST"))
emnist_button.pack(side="left", padx=10)

mnist_button = tk.Button(model_frame, text="Use MNIST", command=lambda: switch_model("MNIST"))
mnist_button.pack(side="right", padx=10)

# Add buttons for predicting and clearing the canvas
button_frame = tk.Frame(root)
button_frame.pack()

predict_button = tk.Button(button_frame, text="Predict", command=lambda: predict_character(canvas))
predict_button.pack(side="left", padx=10)

clear_button = tk.Button(button_frame, text="Clear", command=lambda: clear_canvas(canvas))
clear_button.pack(side="right", padx=10)

# Make the canvas drawable
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black", width=10)

canvas.bind("<B1-Motion>", draw)

# Start the Tkinter main loop
root.mainloop()
