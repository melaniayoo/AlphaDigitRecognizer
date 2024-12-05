import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import messagebox
import torch.nn as nn

# Define image size
IMAGE_SIZE = 16  

# Define model (Use the EMNIST model, can be overridden with a parameter)
class CNN(nn.Module):
    def __init__(self, out_1=16, out_2=32, num_classes=26):  # Adjust num_classes based on dataset
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, num_classes)
        
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

# Tkinter GUI for drawing
class DrawingApp:
    def __init__(self, master, model, dataset_choice):
        self.master = master
        self.model = model  # Use the passed model
        self.dataset_choice = dataset_choice  # Track dataset choice (EMNIST/MNIST)

        self.canvas = tk.Canvas(master, width=IMAGE_SIZE*20, height=IMAGE_SIZE*20, bg='white')
        self.canvas.pack()
        
        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.prediction_label = tk.Label(master, text="Prediction: None", font=("Helvetica", 16))
        self.prediction_label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image_data = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)  # Initialize empty image data

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_data.fill(0)
        self.prediction_label.config(text="Prediction: None")

    def paint(self, event):
        x1, y1 = (event.x // 20) * 20, (event.y // 20) * 20
        x2, y2 = x1 + 20, y1 + 20
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=0)
        self.image_data[y1//20:y2//20, x1//20:x2//20] = 255  # Mark pixels as drawn

    def label_to_char(self, label):
        if self.dataset_choice == 'EMNIST':
            return chr(label + ord('A'))  # Map 0–25 to A–Z
        elif self.dataset_choice == 'MNIST':
            return str(label)  # Map 0–9 to digits
        else:
            return str(label)

    def predict(self):
        # Convert the numpy image array to a PIL image
        image = Image.fromarray(self.image_data)
        composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
        image = composed(image)  # Apply transformations

        # Add a batch dimension and send the image through the model
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            _, predicted_class = torch.max(output.data, 1)

        # Map label to character
        predicted_char = self.label_to_char(predicted_class.item())
        
        # Show prediction result
        messagebox.showinfo("Prediction", f"Predicted: {predicted_char}")
