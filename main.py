import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image1, image2, title1, title2):
    # Create a figure with 1 row and 2 columns for side-by-side images
    plt.figure(figsize=(10, 5))

    # Plot the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first image
    plt.imshow(image1, cmap='gray')
    plt.title(title1)
    plt.axis('off')  # Turn off axis for better visualization

    # Plot the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second image
    plt.imshow(image2, cmap='gray')
    plt.title(title2)
    plt.axis('off')  # Turn off axis for better visualization

    # Display the images
    plt.show()

# Load the image (replace 'image_path' with your image file path)
image = cv2.imread("/Users/OWNER/SideProjects/HandwritingRecognition/Database/train/A.jpeg")
if image is None:
    print("Error: Image not found!")
else:
    # 1. Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Min pixel value:", gray_image.min())
    print("Max pixel value:", gray_image.max())


    ret, new_image = cv2.threshold(image,127, 255, cv2.THRESH_BINARY)
    #plt.imshow(new_image, cmap='gray')
    plot_image(image, new_image, "Original", "Image After Thresholding")
  #  plt.show()



'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Single neuron for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''
'''def load_data(data_dir, labels_file):
    images = []
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split('\t')
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 32))  # Resize to model input dimensions
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(label)
    return np.array(images), labels
'''