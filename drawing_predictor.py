import tkinter as tk
from tensorflow.keras.models import load_model
from PIL import ImageGrab, Image
import numpy as np
import time

# Load the trained model
model = load_model("cnn_emnist_alphanumeric_model.keras")

# Mapping for EMNIST letters (assuming 'A' starts from 0 in your mapping)
def label_to_char(label):
    return chr(label + ord('A'))

# Function to preprocess the canvas image
def preprocess_image(canvas):
    canvas.update()
    # Grab the canvas content as an image
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
    # Resize to 28x28, the input size for your model
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    # Invert colors (black background, white foreground)
    image_array = 255 - image_array
    # Normalize and reshape for the model input
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return image_array

def predict_character(canvas):
    # Preprocess the drawn image
    image = preprocess_image(canvas)
    # Predict the probabilities for each label
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Confidence percentage

    # Map label to character
    predicted_char = label_to_char(predicted_label)
    result_label.config(text=f"Predicted: {predicted_char} (Confidence: {confidence:.2f}%)")

# Function to clear the canvas
def clear_canvas(canvas):
    canvas.delete("all")

# Create the Tkinter GUI
root = tk.Tk()
root.title("Handwritten Character Recognition")

# Create a canvas for drawing
canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack()

# Add result display
result_label = tk.Label(root, text="Predicted: ", font=("Helvetica", 16))
result_label.pack()

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
