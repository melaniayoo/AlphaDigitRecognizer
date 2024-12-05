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
    def __init__(self, out_1=16, out_2=32, num_classes=26):
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
        self.model = model
        self.dataset_choice = dataset_choice

        self.canvas = tk.Canvas(master, width=IMAGE_SIZE*20, height=IMAGE_SIZE*20, bg='white')
        self.canvas.pack()
        
        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.prediction_label = tk.Label(master, text="Prediction: None", font=("Helvetica", 16))
        self.prediction_label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_position)
        self.image_data = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image_data.fill(0)
        self.prediction_label.config(text="Prediction: None")
        self.last_x, self.last_y = None, None  # Reset last mouse position

    def reset_last_position(self, event):
        self.last_x, self.last_y = None, None

    def paint(self, event):
    # Check if there is a previous mouse position
        if hasattr(self, 'last_x') and self.last_x is not None:
            # Draw a smooth line from the last position to the current position
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                fill="black", width=5, capstyle=tk.ROUND, smooth=True
            )

            # Update the corresponding pixels in image_data
            x1, y1 = self.last_x // 20, self.last_y // 20
            x2, y2 = event.x // 20, event.y // 20
            self.image_data[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1] = 255

        # Update the last mouse position to the current one
        self.last_x, self.last_y = event.x, event.y


    def label_to_char(self, label):
        if self.dataset_choice == 'EMNIST':
            return chr(label + ord('A'))
        elif self.dataset_choice == 'MNIST':
            return str(label)
        else:
            raise ValueError(f"Unsupported dataset choice: {self.dataset_choice}")

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please check your configuration.")
            return

        self.prediction_label.config(text="Processing...")
        self.master.update()

        image = Image.fromarray(self.image_data)
        composed = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = composed(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)
            _, predicted_class = torch.max(output.data, 1)

        predicted_char = self.label_to_char(predicted_class.item())
        self.prediction_label.config(text=f"Prediction: {predicted_char}")
        messagebox.showinfo("Prediction", f"Predicted: {predicted_char}")
