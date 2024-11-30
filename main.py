import os
import cv2
import numpy as np

def load_data(data_dir, labels_file):
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
